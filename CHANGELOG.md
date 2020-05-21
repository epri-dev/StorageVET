# Changelog
Notable changes to StorageVET 2.0 are documented in this CHANGELOG.md.

Questions and feedback can be submitted to the Electric Power Research Institute (EPRI) from the Survey Monkey feedback linked on the StorageVET website (https://www.storagevet.com/).

The format is based on [Keep a Changelog] (https://keepachangelog.com/en/1.0.0/).

## [1.0.2] - 2020-04-29 from 2020-02-24
### Added
- added CHANGELOG.md to report on modifications between releases
- added comments to DCM
- added new nsr_max_ramp_rate, sr_max_ramp_rate, fr_response_time, fr_max_ramp_rate to mod
- added response and start up time to load (set to 0)
- added check that requires energy market when including ancillary services
- added self.startup_time attributes for each Technology in technology class
- added non-controllable load
- added data growth/removal helper function; removed separate_constraints attribute from Scenario
- added fill_and_drop_extra_data, add/removes data for analsys and creates optimization levels and initializes degredation iff battery is initalized
- added data growth/removal helper function
- added calc_cba method that calculates all financial outputs
- added nsr_response_time, sr_response_time attributes in BatteryTech, ICE, CAESTech, CurtailPV
- added version to model parameter template name

### Changed
- changed Model_Paramters_Template to allow for 0 min response time / startup_time
- modifed Schema.xml, Model_Parameters_Template_DER.xml, Model_Parameters_DER.csv to reflect attributes changes
- replaced 'Original Net Load' with 'Total Load' in Results post-opt calculations
- changed RA to find events per year, in addition to the mode set by the user
- changed technology to aggregate the state of energy of each ESS in the system
- derate based on 'usuable' energy capcity instead of rated energy capacity
- collecting total SOE in results output
- moved check_for_deferral_failure into deferral
- moved monthly bill calculations into Finances
- moved calc_retail_tariff into finances
- changed Params to read in referenced data before case building

### Removed
- commented out statements in DCM
- removed nsr_response_time, sr_response_time from NSR, SR classes
- removed POI need for startup_time
- removed mask argument for getting the effective load
- removed data headers strings of any extra spaces
- removed artifacts of scenario's former power_kw attribute
- updated SchemaDER.xml, Model_Parameters_Templates_DER.xml, Model_Parameters_Template_DER.csv to include startup_time attributes for technology objects
- removed extra write to error_log when input still continues on to be validated
- removed opt_agg from Battery initialization
- removed extra write to error_log when input still continues on to be validated
- removed separate_constraints attribute from Scenario
- remove timeseries data that is added to PV gen for year that analysis will not run
- removed datafiles from attributes of scenario
- removed all methods that are not used in Params

### Fixed
- generalized children of DER classes to inherit the startup_time attribute from the DER class
- completed testing of controllable load
- fixed RA validation check error and DR reporting error
- fixed multi-year post opt analysis bug
