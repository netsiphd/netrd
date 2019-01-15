"""
<file_name>.py
--------------

Graph distance based on <link to paper, or website, or repository>.

author: <your name here>
email: <name at server dot com> (optional)
Submitted as part of the 2019 NetSI Collabathon.

"""

from .base import BaseDistance


class <AlgorithmName>(BaseDistance):
    def dist(self, G1, G2):
        """A brief one-line description of the algorithm goes here.

        A short paragraph may follow. The paragraph may include $latex$ by
        enclosing it in dollar signs $\textbf{like this}$.

        Params
        ------

        G1, G2 (nx.Graph): two networkx graphs to be compared.

        Returns
        -------

        dist (float): the distance between G1 and G2.

        """
        # Your code goes here!
        # Your code goes here!
        # Your code goes here!

        # Make sure to always save the final distance value inside the
        # self.results dict as follows:
        self.results['dist'] = dist

        # If there are other important quantities that may be of value to
        # the user, you can (and should) also store them in the
        # self.results dict. For example, if the adjacency matrix of
        # the graphs G1 and G2 was computed, we may do something like the
        # following:
        # self.results['adj1'] = first_adjacency_matrix
        # self.results['adj2'] = second_adjacency_matrix

        # The last line MUST be the following. Please make sure that you
        # are only returning one numerical value, the distance between G1
        # and G2. All other values that may be of importance should be
        # stored in self.results instead.
        return dist



### Auxiliary functions go here!
### Auxiliary functions go here!
### Auxiliary functions go here!

# def auxiliary_function1(param1, param2):
#     """Brief description.
#
#     Params
#     ------
#
#     param1 (type): description.
#
#     param2 (type): description.
#
#     Returns
#     -------
#
#     some_value (type) with description.
#
#     """
#     pass
