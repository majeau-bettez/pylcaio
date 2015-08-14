import ArdaInventory
import unittest
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse
import sys
import pdb
import pandas.util.testing as pdt

sys.path.append('/home/bill/software/Python/Modules/')
import matlab_tools as mlt
import matrix_view as mtv


class TestArdaInventory(unittest.TestCase):


    def setUp(self):

        self.matdict = {}

        # A 2X2 FOREGROUND

        self.matdict['PRO_f'] = np.array([['s+orm', 10001, 'kg'],
                                          ['Batt Packing', 10002, 'kg']],
                                          dtype=object)

        self.matdict['PRO_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])

        self.matdict['A_ff'] = scipy.sparse.csc_matrix([[0, 1],
                                                        [10, 11]])

        # WITH THREE STRESSORS

        self.matdict['STR'] = np.array([['stress01', 1614, 'kg'],
                                        ['stress02', 1615, 'kg'],
                                        ['stress03', 1616, 'kg']], dtype=object)


        self.matdict['F_f'] = scipy.sparse.csc_matrix([[0.3, 0.0],
                                                       [0.1, 0.2],
                                                       [0.0, 0.0]])

        self.matdict['y_f'] = scipy.sparse.csc_matrix([[1],[0]])

        # FOUR BACKGROUND PROCESSES

        self.matdict['A_gen'] = scipy.sparse.csc_matrix([[0, 1, 0, 0],
                                                   [0, 0, 2, 0],
                                                   [1, 0, 1, 0],
                                                   [0, 3, 0, 0]])

        self.matdict['PRO_gen'] = np.array([['back01', 1, 'kg'],
                                            ['back02', 2, 'kg'],
                                            ['back03', 3, 'MJ'],
                                            ['back04', 4, 'MJ']], dtype=object)


        self.matdict['A_bf'] = scipy.sparse.csc_matrix([[0, 1],
                                                        [0, 0],
                                                        [1, 0],
                                                        [0, 0]])

        self.matdict['F_gen'] = scipy.sparse.csc_matrix(np.zeros((3,4)))

        self.matdict['C'] = scipy.sparse.csc_matrix(np.ones((1,3)))

        self.matdict['IMP'] = np.array([['GWP100', 1,'kgCO2-eq']], dtype=object)

