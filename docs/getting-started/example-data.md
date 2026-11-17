# Example data

Most pages of this documentation use a VLA HI datacube of the nearby spiral galaxy **NGC 2403**, smoothed to a
beam of 60 arcsec: 150 × 150 pixels of 16 arcsec and 62 channels of 5.2 km/s.

```bash
mkdir bbarolo_examples && cd bbarolo_examples
curl -LO https://github.com/editeodoro/bbarolo2/raw/main/examples/ngc2403.fits
```

The folder [`examples/`](https://github.com/editeodoro/bbarolo2/tree/main/examples) of the repository has
this cube and other datasets, each with a parameter file, described in {doc}`../examples/index`:

| File | Data | Size | Example |
|---|---|---|---|
| [ngc2403.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/ngc2403.fits) | HI cube of NGC 2403, VLA | 5.4 MB | {doc}`../examples/ngc2403` |
| [NGC5633_V500.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/NGC5633_V500.fits) | optical IFU cube of NGC 5633 (Hα + [NII]), CALIFA | 2.3 MB | {doc}`../examples/ngc5633` |
| [pgc53639.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/pgc53639.fits) | HI cube of PGC 53639, ATCA (LVHIS) | 3.5 MB | {doc}`../examples/pgc53639` |
| [ugc1281.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/ugc1281.fits) | HI cube of UGC 1281, WSRT (WHISP) | 0.7 MB | {doc}`../examples/ugc1281` |
| [ngc891.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/ngc891.fits) | HI cube of NGC 891, WSRT (HALOGAS) | 6.0 MB | {doc}`../examples/ngc891` |
| [U3_25160_Ha.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/U3_25160_Ha.fits) | Hα cube of a galaxy at z = 0.9, KMOS (KROSS) | 0.2 MB | {doc}`../examples/u3-25160` |
| [HD142527_13CO.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/HD142527_13CO.fits) | ¹³CO cube of the disk of HD 142527, ALMA | 5.6 MB | {doc}`../examples/hd142527` |
| [2masxj1022_SALT.fits](https://github.com/editeodoro/bbarolo2/raw/main/examples/2masxj1022_SALT.fits) | long-slit spectrum of 2MASX J1022 (Hα + [NII]), SALT | 5.8 MB | {doc}`../examples/2masxj1022` |

The parameter files are next to the data, with the names of the examples (e.g. `ngc2403_3DFIT.par`); run them from
the folder that contains the FITS file, e.g. `BBarolo -p ngc2403_3DFIT.par`. With a copy of the repository
(`git clone https://github.com/editeodoro/bbarolo2`), they are all in `bbarolo2/examples/`.
