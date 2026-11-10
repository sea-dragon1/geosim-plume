# Upstream provenance and patch bases

## Source

- Author: Taras Gerya
- Code: I2VIS
- OSF project: *Slab segmentation*
- DOI: <https://doi.org/10.17605/OSF.IO/BNVTH>
- Archive used as base: `codes.zip` from the OSF record
- Extracted path: `codes/codes/`
- Archive downloaded: 2026-11-07

The OSF archive does not expose a git commit hash. The SHA256 hashes below
define the exact base used to generate the two patches.

## Files required to build

| File | Base SHA256 |
|---|---|
| `gausfast.c` | `f69684752a4916bfa7f9c87097bfcf4dfce1c04ee7b09a6c2d592d7704d09290` |
| `headfast.c` | `b29b43ff5ce7350698868c05e85625413bc9f2d357a218629431a0af10ce2f59` |
| `headjslab.c` | `7c1dacfa71fbeb60e6f48c597592be9b081ebc1388862609b60a2d3d59f631f7` |
| `in2fast.c` | `44f15a7cf4f4f4b8fdd3a99817f09b5d61b112215b8addd6a1edfa26c91c8caa` |
| `loadjslab.c` | `f2b8ac0ec9537d82032495c5a68bbe8680ca02674ccd2888e17d1e6066a1d25d` |
| `loadslab.c` | `4986712ff478c550e786c52f480199af2bd0a1168bcaf8ba6dcc74b4d66a64e5` |
| `markslab.c` | `62cc8af11e694cf0d73859af00413aeb6ea9e084210a3f7973019249b01b5159` |
| `movefast.c` | `304b3857c93cc18ae8ca8157fca34832d918551edf60c337322644682a6b67e6` |
| `moveslab.c` | `f947ca6ba53abb54b48b8100cc38f0934a528511b2003bea85526f319bcef528` |
| `gausslab.c` | `75e80de6d2845544a442cba23ffa5d5d65f233ea997986b5b36fc7998adf741d` |
| `heatfast.c` | `8d45d9454d1bcdf96dacadd6e7cb5782cfe1db4d8f87314e84138a5071aaf678` |
| `heatslab.c` | `fd4477342a0b06cadbffe21900f8a8bae5bbc66a6c8d064d472fa0b2f015191b` |
| `i2fast.c` | `cdae7f7f3e2dd9cfa7be7b5800eb5d6f640aee73300c93ee25efb499ade7720d` |
| `i2jslab.c` | `26edc020ac2a01c56156b443459de7eced40dc168ba3d5943da5d0b574b89657` |

`UPSTREAM.sha256` is the machine-readable form used by
`scripts/apply_patch.sh`.

## Patched file hashes

| File | Plume1 patched SHA256 | Plume2 patched SHA256 |
|---|---|---|
| `gausfast.c` | `ae4aa7c1ecfcb0c65516d432478504f9efb2b8bd730cf3b1246ef41fece5dbf7` | `5af316ab69f5ba8d1dd8e337546b4911fbd76cef711413bf88c324973068f301` |
| `headfast.c` | `a43aee3435e02a98fdf3f0eadd5038f3b65139d5fabd761b07d095415887319b` | same |
| `headjslab.c` | `278607511ad61ccf0b0a1410c49137b64d1e7fac7fbced81fc6e29ea762391d6` | same |
| `in2fast.c` | `158a2484cdd0f64de18e4e9df2d6d3ad7f0ff3b14734d7e840df09f536bbc333` | `edb48646bbe62dda8cb6a82b9fa4c27931b151637a4e63a0c0a18cfbdf3847e6` |
| `loadjslab.c` | `61ac16c04b61008e0ebef6cf016e34836f2c16f3a4072e1d493fa63cf5bf04ef` | same |
| `loadslab.c` | `9c010f51c615cf24e80024e841c6ebad47afc55e6c139d953a47b7137013a15f` | same |
| `markslab.c` | `bca8c7efc94fd3ed4ac0defb7bea1760a36ef0667ed2ba7e9c03248d3142686a` | same |
| `movefast.c` | `cc94add5960d101d42a4cdd040f29c6d129cd5e20720fde2e8a0ae73658c893b` | `d835e6431d48ac90c45ec645a54d784b3c58639a976136c7bef12b928b5c379b` |
| `moveslab.c` | unchanged from base | `f4adf1a8d15c276c33393db9e2635dc3bf9001296c0d04bfd1473d2dcb410266` |

`PATCHED.plume1.sha256` and `PATCHED.plume2.sha256` are the
machine-readable forms used by `scripts/apply_patch.sh`.

## Branch provenance

- Plume1 source copy: `code_read/i2vis_plume1/`
  - eight modified C files;
  - `moveslab.c` matches upstream;
  - build: `-O3 -xCORE-AVX512`.
- Plume2 source copy: `code_read/i2vis_source_plume2/`
  - nine modified C files;
  - build: `-O2 -qopenmp`.

The two branches differ in `gausfast.c` (Plume1 includes `<stdio.h>`),
`in2fast.c` (comment formatting of the disabled melt block),
`movefast.c` (extra diagnostic prints in Plume2), and `moveslab.c`
(diagnostic prints only in Plume2). They also use different `init.t3c`,
`mode.t3c`, and `Makefile` files, supplied under `config/plume1/` and
`config/plume2/`.

## Base-archive contents not used

The upstream archive also contains `movefast2.c`, `i2jfast.c`, plotting
scripts, and example `.t3c` files. The build uses `i2fast.c`,
`in2fast.c`, and `i2jslab.c`; the other files are not part of these
patches.
