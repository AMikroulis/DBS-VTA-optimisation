This is a snapshot (2024-06-18) of the OSS-DBSv2 package from https://github.com/SFB-ELAINE/OSS-DBSv2.

There is a small modification in main.py to disable the success/fail flag file generation, because it interferes with multithreading.

This folder is included only for convenience, because the main script (contacts_mt.py) relies on OSS-DBSv2 to calculate the VTAs, and it is running it into a subprocess to allow parallelisation.

Since this snapshot is not maintained, it is recommended that you install the original OSS-DBSv2 package and adjust the subprocess calls.

Please cite the original work in the link above if you use this software.
