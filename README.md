# feature-detection

Scripts for tree deection from Nearmap 2D and bld detection (future work).

Do not share or distribute any of the content in this repository.

Copyright 2021 Urbanfinity Pty Ltd

## Dockerised ce_automation script

1. Copy folder to desktop
2. Copy in your 2D images folder matching the below file structure

```
feature-detection
│   │   ce_automation_nearmap3.py
│   │   README.md
|   │   environment.yml
└───2D
    │
    │
    │
    └───Tile 1
    |    │   NearMaps_2D_tile_GTiff.tif
    |    │   NearMaps_2D_tile.jpeg
    |
    └───Tile 2
    |    │   NearMaps_2D_tile_GTiff.tif
    |    │   NearMaps_2D_tile.jpeg
    |
    └───Tile (n)
         │   NearMaps_2D_tile_GTiff.tif
         │   NearMaps_2D_tile.jpeg

```

3. Open in vs-code
4. Click "reopen in container" when prompted
5. Run -> Run Without Debugging (^F5)

### Debugging
# SungKimCE
