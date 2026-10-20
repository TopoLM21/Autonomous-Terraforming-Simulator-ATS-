# Offline Qattara DEM

`qattara_etopo1.npz` contains real NOAA **ETOPO1 Ice Surface** elevations and
bathymetry: 28–32° N, 25–30° E, 241 × 301 nodes, native 1 arc-minute spacing.
Rows run south to north, columns west to east. Heights are metres above mean
sea level (MSL), horizontal datum WGS84. Negative height can mean a dry
depression or seafloor; the simulator does not infer water from its sign.

Source (retrieved 2026-10-20):
<https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv?altitude[(28):1:(32)][(25):1:(30)]>

Metadata: <https://coastwatch.pfeg.noaa.gov/erddap/info/etopo180/index.html>.
Citation: Amante, C. and B. W. Eakins (2009), *ETOPO1 1 Arc-Minute Global Relief
Model: Procedures, Data Sources and Analysis*, NOAA Technical Memorandum NESDIS
NGDC-24. DOI: <https://doi.org/10.7289/V5C8276M>.

NOAA permits free use and redistribution. Data is supplied without warranty and
is not intended for legal use. Exact source URL, citation and raw CSV SHA256 are
also embedded in the NPZ metadata. Importing this bundled file requires no
network access and no GIS library.

The source spacing is about 1.6 × 1.85 km here. A 0.25 m work grid interpolates
these observations; it does **not** make the source survey more accurate. Use a
higher-resolution local DEM for detailed work or engineering quantities.
