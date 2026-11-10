# Notice on upstream code and licensing

This package is a **patch** against the I2VIS code released by Taras Gerya
on OSF (<https://doi.org/10.17605/OSF.IO/BNVTH>). It does not contain the
unmodified upstream source tree and does not claim authorship of I2VIS.

The upstream OSF archive does not include a `LICENSE` file. The patches and
case configurations are included with the permission of the I2VIS author
(T. Gerya), confirmed by email on 10 November 2026. No license is asserted
here over upstream I2VIS code, and this package does not grant permission to
redistribute, relicense, or commercially use the upstream code. Users must
obtain I2VIS from the OSF record and comply with the terms applicable to
that release.

The case configurations (`config/init.t3c`, `config/mode.t3c`) and the
build file (`config/Makefile`) are specific to the GeoSim plume setup and
are included to make the modifications reproducible. The thermodynamic
database files are not included; see `database/README.md`.

The scientific interpretation and any errors introduced by the
modifications are the responsibility of the GeoSim authors, not the I2VIS
authors.
