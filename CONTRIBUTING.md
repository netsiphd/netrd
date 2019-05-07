# netrd

Welcome to `netrd` and thanks for your interest in contributing! During development please
make sure to keep the following checklists handy. They contain a summary of all
the important steps you need to take to contribute to the package. As a general statement,
the more familiar you already are with git(hub), the less relevant the detailed instructions
below will be for you.

For an introduction and overview of the project, check out these
[slides](https://docs.google.com/presentation/d/1nnGAttVH5sjzqzHJBIirBSyhbK9t2BdaU6kHaTGdgtM/edit?usp=sharing).


## Types of Contribution

There are multiple ways to contribute to `netrd` (borrowed below list from [here](https://github.com/uzhdag/pathpy/blob/master/CONTRIBUTING.rst)):

#### Report Bugs

To report a bug in the package, open an issue at https://github.com/netsiphd/netrd/issues.

Please include in your bug report:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

#### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

#### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

#### Submit Feedback

The best way to send feedback is to open an issue at https://github.com/netsiphd/netrd/issues.

If you are proposing to implement a distance metric, reconstruction algorithm, or dynamical process, 
see more details below. 

If you are proposing a feature not directly related to implementing a new method:

* Explain in detail why the feature is desirable and how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that your contributions
  are welcome!

##### A Brief Note On Licensing
Often, python code for an algorithm of interest already exists. In the interest of avoiding repeated reinvention of the wheel, we welcome code from other sources being integrated into `netrd`. If you are doing this, we ask that you be explicit and transparent about where the code came from and which license it is released under. The safest thing to do is copy the license from the original code into the header documentation of your file. For reference, this software is [licensed under MIT](https://github.com/tlarock/netrd/blob/master/LICENSE).

## Setup
Before starting your contribution, you need to complete the following instructions once.
The goal of this process is to fork, download and install the latest version of `netrd`.

1. Log in to GitHub.

2. Fork this repository by pressing 'Fork' at the top right of this
   page. This will lead you to 'github.com/<your_account>/netrd'. We refer
   to this as your personal fork (or just 'your fork'), as opposed to this repository
   (github.com/netsiphd/netrd), which we refer to as the 'upstream repository'.

3. Clone your fork to your machine by opening a console and doing

   ```
   git clone https://github.com/<your_account>/netrd.git
   ```

   Make sure to clone your fork, not the upstream repo. This will create a
   directory called 'netrd/'. Navigate to it and execute

   ```
   git remote add upstream https://github.com/netsiphd/netrd.git
   ```

   In this way, your machine will know of both your fork (which git calls
   `origin`) and the upstream repository (`upstream`).

4. During development, you will probably want to play around with your
   code. For this, you need to install the `netrd` package and have it
   reflect your changes as you go along. For this, open the console and
   navigate to the `netrd/` directory, and execute

	```
	pip install -e .
	```

	From now on, you can open a Jupyter notebook, ipython console, or your
    favorite IDE from anywhere in your computer and type `import netrd`.


These steps need to be taken only once. Now anything you do in the `netrd/`
directory in your machine can be `push`ed into your fork. Once it is in
your fork you can then request one of the organizers to `pull` from your
fork into the upstream repository (by submitting a 'pull request'). More on this later!


## Before you start coding

Once you have completed the above steps, you are ready to choose an algorithm to implement and begin coding.

1. Choose which algorithm you are interested in working on.

2. Open an issue at https://github.com/netsiphd/netrd/issues by clicking the "New Issue" button. 

	* Title the issue "Implement XYZ method", where XYZ method is a shorthand name for the distance, reconstruction or dynamics method you plan to implement.
	* Leave a comment that includes a brief motivation for why you want to see this method in `netrd`, as well as any key citations.
	* If such an issue already exists for the method you are going to write, it is a good idea to leave a comment letting others know you are going to work on it.

2. In your machine, create the file where your algorithm is going to
   live. If you chose a distance algorithm, copy
   an existing distance, such as `/netrd/distance/nbd.py`, into 
   `netrd/distance/<algorithm_name>.py`. Similarly, if you chose a reconstruction
   algorithm, copy an existing reconstruction method into 
   `netrd/reconstruction/<algorithm_name>.py`. Please keep in mind that
   <algorithm_name> will be used inside the code, so try to choose
   something that looks "pythonic". In particular, <algorithm_name> cannot
   include spaces, should not include upper case letters, and should use underscores 
   rather than hyphens.

3. Open the newly created file and edit as follows. At the very top you
   will find a string describing the algorithm. Edit this to describe the algorithm you
   are about to code, and preferably include a citation and link to any relevant papers. 
   Also add your name and email address (optional). Do not delete the line 
   `from .base import BaseDistance` or `from .base import BaseReconstructor`. 
   In the next line, change the class name to something appropriate. Guidelines here
   are to use `CamelCaseLikeThis` and not `snake_case_like_this`. (These
   are python guidelines, not ours!)

2. If you are implementing a distance method, you need to edit
   `netrd/distance/__init__.py`. Open it and add the following line:

	```
	from .<your_file_name> import <YourAlgorithmName>
	```

	* Note: there is one dot (.) before <your_file_name>. This is important!
	* Note: this line must go BEFORE the line with `__all__ = []`.

   If you are implementing a reconstruction method, you need to edit
    `netrd/reconstruction/__init__.py` instead, with the same line.

   This line tells the `netrd` package where to find your code.

4. Go back to editing <your_file_name>. After the line that starts with
   `class`, there is a function called `dist` for distances
   and `fit` for reconstructors. This is where the magic happens! There are
   some comments inside of these functions to guide the development, so
   please make sure to read them! You can also use the already existing code
   to get a feel for how you might design your implementation. Feel free to add 
   or remove anything and everything you feel you need. For example, if you 
   need auxiliary functions, you can add those as standalone functions. 
   Please try to base your style on already existing code as much as possible. 
   However, if you really need to do something differently, go ahead and we will 
   discuss how to make it fit with the rest of the package. In particular, 
   the `dist` or `fit` functions _must_ have those names and _must_ receive 
   those parameters (`dist` receives two graphs, while `fit` receives one time series). 
   You can add more parameters if your method needs them, but only _after_ the
   ones already in place. Every time you add a new parameter, it _must_ 
   have a default value.

5. If you need other auxiliary files, we got you covered! Place your
   Jupyter notebooks in the `netrd/notebooks` folder, and any data files
   you may need in the `netrd/data` folder. If you are willing to write a
   short documentation file (this may be plain text, a notebook, latex,
   etc) place that inside `netrd/docs`.
	NOTE: If you want to contribute these files to the git repository,
	please be mindful of the size and format of the files (especially
	when considering uploading data).


## After you finish coding

1. After updating your local code in the previous step, the first thing to
   do is tell git which files you have been working on. (This is called
   staging.) If you worked on a distance algorithm, for example, do

   ```
   git add netrd/distance/<your_file> netrd/distance/__init__.py
   ```

2. Next tell git to commit (or save) your changes:

	```
	git commit -m 'Write a commit message here. This will be public and
	should be descriptive of the work you have done. Please be as explicit
	as possible, but at least make sure to include the name of the method
	you implemented. For example, the commit message may be: add
	implementation of SomeMethod, based on SomeAuthor and/or SomeCode.'
	```

3. Now you have to tell git to do two things. First, `pull` the latest changes from
   the upstream repository (in case someone made changes while you were coding), 
   then `push` your changes and the updated code from your machine to your fork:

	```
	git pull upstream master
	git push origin master
	```

	NOTE: If you edited already existing files, the `pull` may result in
	conflicts that must be merged. If you run in to trouble here, ask
	for help!

4. Finally, you need to tell this (the upstream) repository to include your
   contributions. For this, we use the GitHub web interface. At the top of
   this page, there is a 'New Pull Request' button. Click on it, and it
   will take you to a page titled 'Compare Changes'. Right below the title,
   click on the blue text that reads 'compare across forks'. This will show
   four buttons. Make sure that the first button reads 'base fork:
   netsiphd/netrd', the second button reads 'base: master', the third
   button reads 'head fork: <your_username>/netrd', and the fourth button
   reads 'compare: master'. (If everything has gone according to plan, the
   only button you should have to change is the third one - make sure you
   find your username, not someone else's.) After you find your username,
   GitHub will show a rundown of the differences that you are adding to the
   upstream repository, so you will be able to see what changes you are
   contributing. If everything looks correct, press 'Create Pull
   Request'.
	NOTE: Advanced git users may want to develop on branches other
	than master on their fork. That is totally fine, we won't know
	the difference in the end anyway.


That's it! After you've completed these steps, maintainers will be notified 
and will review your code and changes to make sure that everything is in place. 
Some automated tests will also run in the background to make sure that your 
code can be imported correctly and other sanity checks. Once that is all done, 
one of us will either accept your Pull Request, or leave a message requesting some
changes (you will receive an email either way).
