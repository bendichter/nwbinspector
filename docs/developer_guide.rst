Developer Guide
===============

There are many ways to contribute to the NWBInspector!

Please always begin this process by :nwbinspector-issues:`submitting an Issue ticket <>` on the main repository so we can
openly discuss it taking action. Please do not open a Pull Request (PR) until the Issue has been approved by the team.

The most common contribution is to help us add a new Best Practices and check functions for them. A guideline on how to
build a new data interface and a graphical overview of the object structure can be found in our primary
:nwbinspector-contributing:`Contributing <>` page.

Otherwise feel free to raise a bug report, documentation mistake, or general feature request for our maintainers to address!



.. _adding_custom_checks:

Adding Custom Checks to the Registry
------------------------------------

If you are writing an extension, or have any personal Best Practices specific to your lab, you can incorporate these
into your own usage of the NWBInspector. To add a custom check to your default registry, all you have to do is wrap
your check function with the :py:class:`~nwbinspector.register_checks.register_check` decorator like so...

.. code-block:: python

    from nwbinspector.register_checks import available_checks, register_check, Importance

    @register_check(importance=Importance.SOME_IMPORTANCE_LEVEL, neurodata_type=some_neurodata_type)
    def check_personal_practice(...):
        ...

Then, all that is needed for this to be automatically included when you run the inspector through the CLI is to specify
the modules flag ``-m`` or ``--modules`` along with the name of your module that contains the custom check. If using
the library instead, you need only import the ``available_checks`` global variable from your own submodules, or
otherwise import your check functions after importing the ``nwbinspector`` in your ``__init__.py``.
