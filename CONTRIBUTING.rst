.. highlight:: shell

.. _chapter_contributing:

===============================================================================
Contributing
===============================================================================

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

.. _ml_qm: https://code.roche.com/SMDD/python/ml_qm_n/
.. _ml_qm Issues: https://code.roche.com/SMDD/python/ml_qm_n/-/issues
.. _ml_qm Merge Request: https://code.roche.com/SMDD/python/ml_qm_n/-/merge_requests

You can contribute in many ways:

Types of Contributions
-------------------------------------------------------------------------------

Report Bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Report bugs at `ml_qm Issues`_ in GitLab.

If you are reporting a bug, please include:

* Your operating system name and version.
* The version of the ``ml_qm`` package you are using.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Write Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ml_qm`_ could always use more documentation, whether as part of the
official ``ml_qm`` docs, in docstrings, or even on the web in blog posts,
articles, and such.


Submit Feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at `ml_qm Issues`_ .

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

Get Started!
-------------------------------------------------------------------------------

Ready to contribute? Here's how to set up `ml_qm`_ for local development.

1. Clone `ml_qm`_ repository from GitLab for local development:

   .. code-block:: console

      > git clone git@ssh.code.roche.com:SMDD/python/ml_qm_n.git
      > cd ml_qm_n

2. Setup your environment for Python development:

   .. code-block:: console

      > conda env create --name ml_qm_n --file requirements_dev.yaml
      > conda activate ml_qm_n
      (ml_qm_n) >

3. List available tasks:

   .. code-block:: console

      (ml_qm_n) > invoke --list

   will generate the following list of predefined tasks for the package:

   .. literalinclude:: ../docs/text/invoke-list.txt
      :language: console

4. Create a new branch, called 'work', for development and commit changes to this branch

   .. code-block:: console

      > git checkout -b work

5. Commit changes to your 'work' branch. Please add corresponding test and update documentation
   if necessary.

6. Create a merge request (see `ml_qm Merge Request`_)
   for a lead developer who will review your code and merge your branch to master.

.. seealso::

   * `Python Packages with CDD cookiecutter template <https://rochewiki.roche.com/confluence/display/CHEMINFO/Python+Packages+with+CDD+cookiecutter+template>`_
   * `Continuous Deployment using CDD cookiecutter and GitLab <https://rochewiki.roche.com/confluence/display/CHEMINFO/Continuous+Deployment+using++CDD+cookiecutter+and+GitLab>`_
   *  :ref:`chapter_installation` chapter
