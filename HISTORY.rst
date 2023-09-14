.. |repo| replace:: BioEvents
.. |bug| replace:: **Bug Fixes**
.. |feat| replace:: **Features Added**
.. |refactor| replace:: **Notable Refactors**
.. |test| replace:: **Testing**

=======
History
=======

0.2.0 (2023-09-13)
------------------
First effective release of |repo|.
Implements significant changes to repo structure and engineering practices.

|feat|

* Adds core time series event handling code
* Adds code for handling sleep stage time series data as hypnograms
* Includes Jupyter notebook tutorial with docs integration

|test|

* Adds first testing for event_handling and hypnogram
* Includes test utility for simulating hypnogram data

|refactor|

* Removes several irrelevant files
* Uses PDM as dependency manager
* Uses pre-commit with black for code quality
* Implements ReadTheDocs via Sphinx with myst_nb support for tutorial notebook

0.1.0 (2023-08-11)
------------------
Represents only the cookiecutter output.
