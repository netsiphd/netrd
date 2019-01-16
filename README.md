# netrd

Welcome to the NetSI 2019 Collabathon repository page! In order to
contribute to this repository please read carefully the instructions in the
[slides](https://docs.google.com/presentation/d/1R-IS8kUHVhmLnVOs22wLmEV7wVbT4p5gY1s_HBBHRLU/edit?usp=sharing).

That might be a lot to remember, so during development please make sure to
keep the following checklists handy. They contain a summary of all the
steps you need to take (for details, go back to the slides).

For all other information about the project, please visit the
[Mega Doc](https://docs.google.com/document/d/1LMBFgE8F9fR3mZjB9WRDr5bj_R7Z5kcw04K-LWN_cGM/edit?usp=sharing).
Or for a more general introduction, check out these [slides](https://docs.google.com/presentation/d/1nnGAttVH5sjzqzHJBIirBSyhbK9t2BdaU6kHaTGdgtM/edit?usp=sharing). 
Lastly, after writing up a brief description of the method you have
implemented, please include it in this [document](https://v2.overleaf.com/5374841856ppnrdkgqkqpr).

## Setup

These are steps you must take before starting your contribution, and only
need to do them once.

1. Log in to GitHub.

2. Fork this repository by pressing 'Fork' at the top right of this
   page. This will lead you to 'github.com/<your_account>/netrd'. We refer
   to this as your personal fork (or just 'your fork'). This repository
   (github.com/netsiphd/netrd) is the 'upstream repository'.

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
fork into the upstream repository. More on this later!


## Start

Once you're all setup and ready to start coding, these are the steps you need.

1. Choose which algorithm you (or your team) will be working on by
   [signing up here](https://docs.google.com/spreadsheets/d/1N_9_85MjYFYClloKOQkMz-L6g3wuclIRshcIdE7MfTs/edit?usp=sharing).
   In this spreadsheet there are three sheets. The reconstruction
   algorithms are in the 'Time series to networks' sheet; the distance
   algorithms are in the 'Graph distance / network similarity'
   sheet. Choose one algorithm that is marked as 'todo', and write 'doing'
   instead. (The cell should go from red to orange.)

2. In your machine, create the file where your algorithm is going to
   live. If you chose a distance algorithm, copy
   `netrd/distance/template.py` into
   `netrd/distance/<algorithm_name>.py`. If you chose a reconstruction
   algorithm, copy `netrd/reconstruction/template.py` to
   `netrd/reconstruction/<algorithm_name>.py`. Please keep in mind that
   <algorithm_name> will be used inside the code, so try to choose
   something that looks "pythonic". In particular, <algorithm_name> cannot
   include spaces, and it is recommended that it doesn't have upper case
   letters either.

3. Open the newly created file and edit as follows. At the very top you
   will find a string. Please add one or two lines about the algorithm you
   are about to code, and preferably include a link. Add also your name and
   email (optional). Do not delete the line `from .base import
   BaseDistance` or `from .base import BaseReconstructor`. In the next
   line, change the class name to something appropriate. Guidelines here
   are to use `CamelCaseLikeThis` and not `snake_case_like_this`. (These
   are python guidelines, not ours!)

4. Inside the class, there is only one function, `dist` for distances and
   `fit` for reconstructors. This is where the magic happens! There are
   some comments inside of these functions to guide the development, please
   make sure to read them! Feel free to add anything and everything you
   feel you need. For example, if you need auxiliary functions feel free to
   add those as standalone functions. Please try to follow the template as
   much as possible. However, if you really need to do something
   differently, go ahead and we will discuss how to make it fit with the
   rest of the package.

5. If you need other auxiliary files, we got you covered! Place your
   Jupyter notebooks in the `netrd/notebooks` folder, and any data files
   you may need in the `netrd/data` folder. If you are willing to write a
   short documentation file (this may be plain text, a notebook, latex,
   etc) place that inside `netrd/docs`.


## Finish

1. Before adding your contribution to this repository, you need to make
   sure that your code doesn't conflict with other code that other people
   may have written in the meantime. For this, you need to execute
   
   ```
   git pull upstream master
   ```
   
   If this doesn't show any errors, you're good to go! On the other hand,
   the possible errors at this stage may be confusing and hard to fix. If
   you've never done this before, we recommend contacting someone with some
   experience in git to help you at this stage. We do not foresee a lot of
   errors at this step because we have designed the repository in such a
   way that your code should never conflict with anybody else's... but it
   still may happen.

2. If you implemented a distance method, you need to edit
   `netrd/distance/__init__.py`. Open it and add the following line:

	```
	from .<your_file_name> import <YourAlgorithmName>
	```

	Note: there is one dot (.) before <your_file_name>. This is important!
	Note: this line must go BEFORE the line with `__all__ = []`.

	If you implemented a reconstruction method, you need to edit
    `netrd/reconstruction/__init__.py` instead, with the same line.

3. After updating your local code in the previous step, the first thing to
   do is tell git which files you have been working on. (This is called
   staging.) If you worked on a distance algorithm, do

   ```
   git add netrd/distance/<your_file> netrd/distance/__init__.py
   ```

	If you worked on a reconstruction algorithm, do
	
	```
	git add netrd/reconstruction/<your_file> netrd/reconstruction/__init__.py
	```

	At this point you also want to add any other files that you used during
    development (see last step under 'Start'). For example, if you added a
    Jupyter notebook to the `netrd/notebooks` folder then you should
    execute

	```
	git add netrd/notebooks/<your_notebook_name>
	```

    (Note: other tutorials might encourage using `git commit -a` to stage
    modified files while writing a commit message for simplicity. Since this
    project will have a lot of files, and a lot of people working on lots of
    files, we would recommend not using `git commit -a`, and especially not
    using `git add -A`, to avoid staging and committing files you did not intend
    to commit, which could create merge conflicts for other people.)

4. Next tell git to commit (or save) your changes:

	```
	git commit -m 'Write a commit message here. This will be public and
	should be descriptive of the work you have done. Please be as explicit
	as possible, but at least make sure to include the name of the method
	you implemented. For example, the commit message may be: add
	implementation of SomeMethod, based on SomeAuthor and/or SomeCode.'
	```

5. Now you have to tell git to send your changes from your machine to your
   fork:

	```
	git push origin master
	```

6. Finally, you need to tell this (the upstream) repository to include your
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
   contributing. If everything looks correctly, press 'Create Pull
   Request'.

7. That's it! We will be notified and will review your code and changes to
   make sure that everything is in place. Some automated tests will also
   run in the background to make sure that your code can be imported
   correctly and other sanity checks. Once that is all done, one of us will
   either accept your Pull Request, or leave a message requesting some
   changes (you will receive an email either way).

8. Once your code is included in the package, it's time to implement
   another method! You can go back to the sign-in sheet to see what
   algorithms have yet to be implemented. If you start another algorithm,
   you do not need to perform the steps under the 'Setup' heading at the
   top of this page, but please make sure to follow the steps under
   'Start'.
