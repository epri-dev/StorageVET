# Changelog
Notable changes to StorageVET 2.0 are documented in this CHANGELOG.md.

Questions and feedback can be submitted to the Electric Power Research Institute (EPRI) from the Survey Monkey feedback linked on the StorageVET website (https://www.storagevet.com/).

The format is based on [Keep a Changelog] (https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2020-04-29 from 2021-05-31
### Added
- extra print statements/ log reporting around the degredation module in Battery ESS

### Changed
- refactoring that was a result of the CBA module in DER-VET
- enhanced result reporting
- refactors LF and FR to follow DRY coding
- energy price always has the same column name in timeseries CSV result
- DR inputs: DR program start and end times defined with hour beginning

### Removed
- pandas warning as a result of deprecating methods

### Fixed
- proforma bugs: inflation rate application, value fill btw optimization years, back fill of values
- cost benefit module calculation bugs

## [1.0.2] - 2019-06-10 from 2020-04-29
### Added
- added JSON input feature
- added general rotating generator model
- added ability to define more than one of the same 'TAG'

### Changed
- changed schema from XML to JSON
- changed Schema.json, Model_Parameters_DER.csv to reflect attributes changes
- changed the User service to apply constraints at the POI

### Removed
- removed all old "GUI" files

### Fixed

