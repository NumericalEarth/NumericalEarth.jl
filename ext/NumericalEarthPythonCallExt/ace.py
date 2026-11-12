import types

import numpy
import torch
import xarray

import fme.core.device
from fme.ace.stepper.single_module import StepperOverrideConfig, load_stepper
from fme.core.step.args import StepArgs


def julia_tensor(address, shape, strides, typestr, cuda):
    interface = {"shape": shape, "strides": strides, "typestr": typestr, "data": (address, False), "version": 3}
    if cuda:
        return torch.as_tensor(types.SimpleNamespace(__cuda_array_interface__=interface))
    return torch.from_numpy(numpy.asarray(types.SimpleNamespace(__array_interface__=interface)))


class Emulator:
    def __init__(self, stepper, state, forcing):
        self.stepper = stepper
        self.state = state
        self.stepper_state = None
        self.forcing = forcing
        self.forcing_names = (set(stepper.config.input_names) - set(stepper.out_names)) | {"surface_temperature", "sea_ice_fraction", "ocean_fraction"}
        self.next_step_forcing_names = set(stepper.config.next_step_forcing_names)

    def forcing_at(self, time):
        sample = self.forcing[time.year].sel(time=time)
        template = self.state["surface_temperature"]
        return {name: torch.as_tensor(sample[name].values, device=template.device).broadcast_to(template.shape[1:]).unsqueeze(0)
                for name in self.forcing_names}

    def step(self, time, next_time, surface, targets):
        now = self.forcing_at(time)
        later = self.forcing_at(next_time)
        current = {name: later[name] if name in self.next_step_forcing_names else value for name, value in now.items()}

        land = now["land_fraction"]
        temperature, concentration, water_fraction = (surface[name].unsqueeze(0).to(land.dtype)
                                                      for name in ("surface_temperature", "sea_ice_concentration", "water_fraction"))
        # Cells without water in the coupled model take their surface from the forcing files
        has_water = water_fraction > 0
        fractions = {"sea_ice_fraction": torch.where(has_water, (1 - land) * concentration, now["sea_ice_fraction"]),
                     "ocean_fraction": torch.where(has_water, (1 - land) * (1 - concentration), now["ocean_fraction"])}
        water_temperature = torch.where(has_water, temperature, now["surface_temperature"])
        surface_temperature = (1 - land) * water_temperature + land * self.state["surface_temperature"]

        args = StepArgs({**self.state, **current, **fractions, "surface_temperature": surface_temperature},
                        {**later, **fractions}, stepper_state=self.stepper_state)
        with torch.inference_mode():
            result = self.stepper.step(args)

        self.state = {name: result.output[name] for name in self.stepper.prognostic_names}
        self.stepper_state = result.stepper_state
        self.copy_outputs(result.output, targets)

    def copy_outputs(self, output, targets):
        for name, (view, sign) in targets.items():
            view.copy_(sign * output[name][0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def open_emulator(checkpoint_path, initial_condition_path, forcing_paths, start_date, force_cpu):
    fme.core.device._FORCE_CPU = force_cpu
    stepper = load_stepper(checkpoint_path, StepperOverrideConfig(ocean=None), use_ema_if_available=True)
    stepper.set_eval()
    device = fme.core.device.get_device()
    initial_condition = xarray.open_dataset(initial_condition_path).sel(time=start_date)
    state = {name: torch.as_tensor(initial_condition[name].values, device=device).unsqueeze(0)
             for name in stepper.prognostic_names}
    forcing = {int(dataset.time.dt.year[0]): dataset for dataset in map(xarray.open_dataset, forcing_paths)}
    return Emulator(stepper, state, forcing), initial_condition.latitude.values, initial_condition.longitude.values
