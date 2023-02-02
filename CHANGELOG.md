# Changelog
Notable changes to StorageVET 2.0 are documented in this CHANGELOG.md.

Questions and feedback can be submitted to the Electric Power Research Institute (EPRI) from the Survey Monkey feedback linked on the StorageVET website (https://www.storagevet.com/).

The format is based on [Keep a Changelog] (https://keepachangelog.com/en/1.0.0/).

## [1.2.3] - 2023-01-26
### Fixed
- simplify the README
- bypass check on valid project start and end year when analysis-horizon-mode is not set to 1
- update the README package installation procedure to align with the dervet README
- Energy Storage 'SOC %' in the output timeseries CSV should have values ranging from 0 to 100 (and not a fraction of 1)
- remove triple equals sign from line in the requirements file
- allow om_cost to end up in the proforma output CSV file for technologies
- handle the timestep input in a more robust way to avoid uncaught code failures
- allow Backup Energy Service to create a minum energy constraint
- allow cases with two different construction-year values to not fail
- do not ignore the Battery auxiliary-load (hp) input
  - add this to the Total Load, and in the agg_power variables for POI constraints
- corrects the DR and RA minimum der dispatch power constraints
  - exclude power from Load and Intermittent-Resource Technologies
### Added
- make reading of model parameters more robust
- attaches 'User Constraints' label to output timeseries user constraint columns
- add SystemRequirementError class
- pytests added
  - for valid timestep frequencies
  - for POI constrsints
  - for Energy constraints
  - for DR minimum power constraints
- make the testing library methods more robust and easier to use
### Changed
- allow the growth of value streams in non-optimization years
  - for DemandResponse, ResourceAdequacy, and UserConstraints
- remove all API keys from this repo
- re-work the way system requirements are applied as constraints
  - to ensure that once created, a system requirement cannot be ignored
- redefine the agg_power variables in POI to include generation from all technologies
  - with both charge and discharge components

## [1.2.2] - 2022-05-05 to 2022-07-07
### Added
- introduce 3 new cvx solver error types:
  - SolverInfeasibleError, SolverUnboundedError and SolverError
  - adds tests for these by reading in bad data
### Changed
- improved handling of input time series data
  - new method: get_single_series()
  - introduce more specific error classes: TimeseriesDataError and TimeseriesMissingError
  - better error and warning messaging and readability
- sets all min and max battery constraint values to empty in the default input time series
- adds column: LF Price ($/kW) to default time series
### Removed
- remove line where all missing/empty time series values are set to zero

## [1.2.1] - 2022-03-31 to 2022-05-04
### Changed
- change in upstream dervet repo only

## [1.2.0] - 2021-09-08 to 2022-03-30
### Added
- pytests to ensure that the default model parameter CSV file runs when input into run_StorageVET.py
- pytests to ensure that with each technology active along with a battery, the default model parameter CSV runs
- adds three new Finance tag inputs to set the fuel price: liquid, gas, other
- allow the forces use of the GLPK_MI solver when necessary
- a copy of the model parameters input CSV is now copied to the Results folder for each run
- adds required rows for all technologies in the default model parameter csv

### Changed
- upgrade supported/recommended Python version to 3.8.13
  - Python package requirements have been updated
  - Update installation instructions: Python environment creation, conda-route, pip-route
- Adds missing udis term in proforma of technologies for variable om cost and fuel cost
- limit MACRS term to no greater than 20
- re-structures how fuel costs are handled
  - fuel costs are now constant over time (time independent)
  - fuel prices are now set in a project-wide manner
  - fuel cost is now determined by the fuel_type within a technology and an associated fuel_price in the Finance tag
  - units of fuel are now relative to Btu and not gallons
  - fuel price is collected as a user input in $/MMBtu but converted into $/kW upon ingest
- changes all kw (lower-case w) labels to kW (upper-case W)
- changes all BTU labels to Btu

### Removed
- remove fuel costs that are tied to a specific technology
  - dropped use of natural gas and diesel labels for fuel

### Fixed
- corrects CAES to bypass degradation module
- add fuel costs to proforma for CAES
- fix bug where CAES was having its tag overwritten
- MarketServiceUp calculates worst case energy based on dt instead of duration
- FR energy throughput now divides by the ES rte in objective function and proforma
- Enable user service to apply min discharge and min charge constraints

## [1.1.3] - 2021-08-04 to 2021-09-07
### Changed
- Changed the expected type to float for yearly_degrade battery input

### Fixed
- a bug that checks infeasibility in User Services timeseries inputs was fixed
- Degradation Fixes
  - State of Health calculation was corrected
  - Application of calendar degradation was corrected
- Simplifies system_requirements infeasibility checks
  - removed place where this being checked redundantly
  - fixed a bug that reports a contributor list to an error

## [1.1.2] - 2021-07-09 to 2021-08-03
### Changed
- The O&M cost methodology in the proforma was improved
  - first escalate o&m rates (usually by inflation)
  - then calculate the cumulative energy dispatch values for optimization years, and use linear interpolation
  - then multiply the rate by the energy each year to get an o&m cost in dollars

### Fixed
- make sure that OM cost of PV gets added to proforma

## [1.1.1] - 2021-05-31 to 2021-07-09
### Changed
- refactor DR input collection for DRY coding
- all growth rates have a minimum value of -100 percent
- Results label can be empty without erroring
- results/simple_monthly_bill.csv will have consistent output showing the billing periods

### Fixed
- growth rate for DA energy prices converted to fraction of 1 at initialization only; same for all growth rates
- corrected the logic and docstrings in Params class bad_active_combo method
- DR inputs bug: length and end_hour consistency errors
- Deferral Proforma fill for in-between years
- corrected UP/Down naming in Market Services
- single billing period bug for a given month was fixed

## [1.1.0] - 2020-04-29 to 2021-05-31
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

## [1.0.2] - 2019-06-10 to 2020-04-29
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

