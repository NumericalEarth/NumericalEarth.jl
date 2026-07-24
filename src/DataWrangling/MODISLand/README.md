# Setting `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` for downloading MODIS (LP DAAC) datasets

The MODIS land products (MCD43 albedo, MCD15 LAI, MCD12Q1 PFT) are served from
NASA's LP DAAC on Earthdata Cloud and require a free **Earthdata Login**.

The first step is to create (or sign in to) an Earthdata Login account:

> https://urs.earthdata.nasa.gov

Use the **username** you chose at registration (not your email address) and your
Earthdata Login **password** — this is a different credential from the ECCO
WebDAV password documented in [`../ECCO/README.md`](../ECCO/README.md).

Next, authorize the LP DAAC application that serves these files: in your profile
go to **Applications → Authorized Apps**, search for and approve

> **LP DAAC Cumulus PROD**

Finally, set your username and password as environment variables, either in a
file (e.g. `~/.zshrc` or `~/.bashrc`):

```bash
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password
```

or within Julia by

```julia
ENV["EARTHDATA_USERNAME"] = "your_username"
ENV["EARTHDATA_PASSWORD"] = "your_password"
```

The download uses these to authenticate against `urs.earthdata.nasa.gov` via a
temporary `.netrc` file that is written to a scratch directory and removed after
use, so your password is never left on disk.
