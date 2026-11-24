Duplicated Data
===============

Data that are stored twice within one file are easy to miss. The file is valid, every object reads correctly, and
the problem only shows up when an analysis counts the same unit twice or treats a copied column as an independent
measurement. Duplicates of this kind usually come from conversion code: a loop that writes the same object on every
iteration, an index that is off by one, or a variable that is reused for two outputs.

The checks on this page compare objects within one file. They first rule out pairs that cannot match from their
shape and data type, then hash what remains. Datasets that are too large to read whole are compared on a few
sampled blocks, and the message says when a result rests on samples. Pairs that are HDF5 links to the same dataset
are not reported, since a link is the right way to make one object available in two places.


.. _best_practice_no_duplicate_units:

Units Stored Twice
~~~~~~~~~~~~~~~~~~

Each row of the :ref:`nwb-schema:sec-Units` table should hold a distinct unit. Two rows with identical spike
trains almost always mean that one unit was written twice, either under the same ``id`` or under two different ids.
An analysis that treats each row as a unit will count it twice.

Check function: :py:meth:`~nwbinspector.checks._duplicates.check_units_identical_spike_trains`


.. _best_practice_no_duplicate_rows:

Repeated Rows and Repeated Times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An event or interval should be stored once. In an event-like table (one with a time column such as
``start_time`` or ``timestamp``), a row that repeats another row in every column is an event stored more than once.
Likewise, an array-valued column whose per-row arrays repeat across rows (for example, a tuning curve per ROI that
is the same for every ROI) usually holds one row's values written into every row. Columns whose rows all hold the
same sorted axis, such as bin centers, are expected to repeat and are not reported.

Times within one row of a ragged time column, such as the ``event_times`` of an event type, and the ``timestamps``
of a :ref:`nwb-schema:sec-TimeSeries` should not repeat. Repeats are reported when they make up at least 1% of all
values, or at least 10% of one row with 100 values or more. Times that are all multiples of 1 ms or coarser are not
checked, since events recorded at that resolution can share a time without being stored twice. Repeated spike times
are reported by :py:meth:`~nwbinspector.checks._ecephys.check_ascending_spike_times` instead.

Check functions: :py:meth:`~nwbinspector.checks._duplicates.check_table_duplicate_rows`,
:py:meth:`~nwbinspector.checks._duplicates.check_table_ragged_times_repeated`,
:py:meth:`~nwbinspector.checks._duplicates.check_timestamps_repeated`


.. _best_practice_no_identical_series:

Identical TimeSeries
~~~~~~~~~~~~~~~~~~~~

Two :ref:`nwb-schema:sec-TimeSeries` in one file should not hold identical data. When the two are of different
kinds, one of them most likely holds the wrong data. This includes a processed series that is identical to a series
in ``acquisition`` (an LFP that is the raw signal, for example), two series of different neurodata types or units,
and two event-like series (constant values at explicit timestamps) that share their timestamps under different names.

When the two series are of the same kind, the second is more likely a redundant copy. If both are meant to hold the
same data, store the data once and link the second series to it, which keeps the file smaller and makes the
relationship explicit. Constant data on a rate-based time base (two empty or all-zero channels, for example) are
not reported.

Check functions: :py:meth:`~nwbinspector.checks._duplicates.check_identical_time_series`,
:py:meth:`~nwbinspector.checks._duplicates.check_redundant_time_series_copies`


.. _best_practice_no_identical_columns:

Identical Columns
~~~~~~~~~~~~~~~~~

Two columns of one table, or two columns of the ``data`` of one 2-D :ref:`nwb-schema:sec-TimeSeries`, should not
hold identical values, for example a ``z`` coordinate that copies ``y`` in the electrodes table or in a position.
Columns with fewer than three distinct values are not compared, nor are pairs of time columns, which often coincide
by design (``stop_time`` and a custom ``return_stop_time``, for example).

For large series, the comparison is limited by how the data are chunked. It reads one block of rows for every
column and then only the columns that match there, and it skips a series whose first block alone would decompress
more than 64 MB of chunks. This keeps the check fast when files are streamed, at the cost of not checking raw
recordings that are chunked across many rows.

Check functions: :py:meth:`~nwbinspector.checks._duplicates.check_table_identical_columns`,
:py:meth:`~nwbinspector.checks._duplicates.check_time_series_identical_columns`
