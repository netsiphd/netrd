# netrd

Welcome to the NetSI 2019 Collabathon repository page! In order to
contribute to this repository please read carefully the instructions in the
[slides](https://docs.google.com/presentation/d/1R-IS8kUHVhmLnVOs22wLmEV7wVbT4p5gY1s_HBBHRLU/edit?usp=sharing).

That might be a lot to remember, so during development please make sure to
keep the following checklists handy. They contain a summary of all the
steps you need to take (for details, go back to the slides).

For all other information about the project, please visit the
[Mega Doc](https://docs.google.com/document/d/1LMBFgE8F9fR3mZjB9WRDr5bj_R7Z5kcw04K-LWN_cGM/edit?usp=sharing).


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
   git clone git@github.com:<your_account>/netrd.git
   ```

   Make sure to clone your fork, not the upstream repo. This will create a
   directory called 'netrd/'. Navigate to it and execute

   ```
   git remote add upstream git@github.com:netsiphd/netrd.git
   ```

   In this way, your machine will know of both your fork (which git calls
   `origin`) and the upstream repository (`upstream`).


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
