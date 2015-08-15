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
                                                        [1, 0],
                                                        [1, 0],
                                                        [0, 0]])

        self.matdict['F_gen'] = scipy.sparse.csc_matrix(np.zeros((3,4)))

        self.matdict['y_gen'] = scipy.sparse.csc_matrix(np.zeros((4,1)))

        self.matdict['C'] = scipy.sparse.csc_matrix(np.ones((1,3)))

        self.matdict['IMP'] = np.array([['GWP100', 1,'kgCO2-eq']], dtype=object)

        #-----------------

        self.bigdict = self.matdict.copy()
        self.bigdict['PRO_gen'] = np.array([['back01', 1, 'kg'],
                                            ['back05', 5, 'kg'],
                                            ['back03', 3, 'MJ'],
                                            ['back02', 2, 'kg'],
                                            ['back04', 4, 'MJ']], dtype=object)

        self.bigdict['A_gen'] = scipy.sparse.csc_matrix(
                                                  [[0, 1, 0, 0, 0],
                                                   [0, 0, 0, 0, 0],
                                                   [1, 0, 1, 0, 0],
                                                   [0, 0, 2, 0, 0],
                                                   [0, 3, 0, 0, 0]])

        self.bigdict['y_gen'] = scipy.sparse.csc_matrix(np.zeros((5, 1)))
        self.bigdict['F_gen'] = scipy.sparse.csc_matrix(np.zeros((3, 5)))

        #-----------------

        self.smalldict = self.matdict.copy()
        self.smalldict['PRO_gen'] = np.array([['back01', 1, 'kg'],
                                            ['back03', 3, 'MJ'],
                                            ['back04', 4, 'MJ']], dtype=object)

        self.smalldict['A_gen'] = scipy.sparse.csc_matrix(
                                                  [[0, 0, 0],
                                                   [1, 1, 0],
                                                   [0, 0, 0]])

        self.smalldict['y_gen'] = scipy.sparse.csc_matrix(np.zeros((3, 1)))
        self.smalldict['F_gen'] = scipy.sparse.csc_matrix(np.zeros((3, 3)))


    def test_import_and_export_matdict_keys_roundtrip(self):

        a = ArdaInventory.ArdaInventory()
        a.extract_background_from_matdict(self.matdict)
        a.extract_foreground_from_matdict(self.matdict)
        matdict = a.export_system_to_matdict()

        assert(matdict.keys() == self.matdict.keys())

    def test_match_foreground_background_trivial(self):
        a = ArdaInventory.ArdaInventory()
        a.extract_background_from_matdict(self.matdict)
        a.extract_foreground_from_matdict(self.matdict)

        b = ArdaInventory.ArdaInventory()
        b.extract_background_from_matdict(self.matdict)
        b.extract_foreground_from_matdict(self.matdict)

        a.match_foreground_to_background()
        assert(np.all(a.A_bf == b.A_bf))
        assert(np.all(a.F_f == b.F_f))

    def test_match_foreground_background(self):
        a = ArdaInventory.ArdaInventory()
        a.extract_foreground_from_matdict(self.matdict)
        a.extract_background_from_matdict(self.bigdict)
        a.match_foreground_to_background()
        assert(np.all(a.A_bf.values == np.array([[0, 1],
                                                 [0, 0], # <--row5 insert here
                                                 [1, 0],
                                                 [1, 0],  # <--row2 moved here
                                                 [0, 0]])))

        
    def test_match_foreground_background_flowlosses(self):
        a = ArdaInventory.ArdaInventory()
        a.extract_foreground_from_matdict(self.matdict)
        a.extract_background_from_matdict(self.smalldict)
        with self.assertRaises(ValueError):
            a.match_foreground_to_background()





if __name__ == '__main__':
        unittest.main()
