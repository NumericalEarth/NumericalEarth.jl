# Setting `ECCO_USERNAME` and `ECCO_WEBDAV_PASSWORD` environment variables for downloading ECCO datasets

The first step is to find the username and password for your "WebDAV/Programmatic API" credentials on NASA's Earthdrive.
For this you have to either login or make an account via the "EARTHDATA login":

> https://urs.earthdata.nasa.gov

Either register and then sign in or, if you are already registered, sign in. Next, navigate to the ECCO drive:

> https://ecco.jpl.nasa.gov/drive/

This should produce a screen similar to the following:

![image](https://github.com/user-attachments/assets/490d9098-aece-4e9c-82d7-3ec86e833347)

showing your WebDAV/Programmatic API credentials -- except in place of the black boxes that say `your_username` and `cRaZYpASSwORD`,
you should see _your_ username and password.
Copy the content of `Username:` to the environment variable `ECCO_USERNAME` and the content of `Password` to `ECCO_WEBDAV_PASSWORD`,
either in a file:

```bash
export ECCO_USERNAME=your_username
export ECCO_WEBDAV_PASSWORD=cRaZYpASSwORD
```

or within Julia by

```julia
ENV["ECCO_USERNAME"] = "your_username"
ENV["ECCO_WEBDAV_PASSWORD"] = "cRaZYpASSwORD"
```

## The `ECCO2` directory is no longer served

`https://ecco.jpl.nasa.gov/drive/files` lists only `NearRealTime`, `Version4`, and `Version5`, and
every path beneath `files/ECCO2/` answers `403 Forbidden` to an account the drive otherwise accepts.
This covers `ECCO2Monthly`, `ECCO2Daily`, `ECCO2DarwinMonthly`, and `ECCO4DarwinMonthly` — both
ECCO-Darwin datasets are stored under `files/ECCO2/` whatever grid their name refers to.
`ECCO4Monthly` reads from `files/Version4/` and is unaffected. Setting the variables above will not
restore access. Downloads are still attempted, so these datasets will work again if the directory
returns, at which point this note and the check in `ECCO.jl` can go.
