import ArdaInventory
import unittest
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse
import sys
import pdb
import pandas.util.testing as pdt
import IPython

sys.path.append('/home/bill/software/Python/Modules/')
import matlab_tools as mlt
import matrix_view as mtv
sys.path.append('/home/bill/software/pymrio/')
import pymrio


class TestArdaInventory(unittest.TestCase):


    def setUp(self):

        self.matdict = {}

        # A 2X2 FOREGROUND

        self.matdict['PRO_f'] = np.array([['s+orm', 10005, 'kg'],
                                          ['Batt Packing', 10002, 'kg']],
                                          dtype=object)

        self.matdict['PRO_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])

        self.matdict['A_ff'] = scipy.sparse.csc_matrix([[0, 1],
                                                        [10, 11]])

        # WITH THREE STRESSORS

        self.matdict['STR_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])
        self.matdict['STR'] = np.array([['stress01', 1614, 'kg'],
                                        ['stress02', 1615, 'kg'],
                                        ['stress03', 1616, 'kg']], dtype=object)


        self.matdict['F_f'] = scipy.sparse.csc_matrix([[0.3, 0.0],
                                                       [0.1, 0.2],
                                                       [0.0, 0.0]])

        self.matdict['y_f'] = scipy.sparse.csc_matrix([[1.0],[0.0]])

        # FOUR BACKGROUND PROCESSES

        self.matdict['A_gen'] = scipy.sparse.csc_matrix([[0, 1, 0, 0],
                                                   [0, 0, 2, 0],
                                                   [1, 0, 1, 0],
                                                   [0, 3, 0, 0]])

        self.matdict['PRO_gen'] = np.array([['back01', 1, 'kg'],
                                            ['back02', 2, 'kg'],
                                            ['back03', 3, 'MJ'],
                                            ['back04', 4, 'MJ']], dtype=object)


        self.matdict['A_bf'] = scipy.sparse.csc_matrix([[0.0, 1.0],
                                                        [1.0, 0.0],
                                                        [1.0, 0.0],
                                                        [0.0, 0.0]])

        self.matdict['F_gen'] = scipy.sparse.csc_matrix(np.zeros((3,4)))

        self.matdict['y_gen'] = scipy.sparse.csc_matrix(np.zeros((4,1)))

        self.matdict['C'] = scipy.sparse.csc_matrix(np.ones((1,3)))

        self.matdict['IMP'] = np.array([['GWP100', 1,'kgCO2-eq']], dtype=object)
        self.matdict['IMP_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])

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


    def test_append_to_foreground(self):
        a = ArdaInventory.ArdaInventory(1)
        b = ArdaInventory.ArdaInventory(1)
        a.extract_background_from_matdict(self.matdict)
        a.extract_foreground_from_matdict(self.matdict)

        B = {}
        B['PRO_f'] = np.array([['foo', 10, 'kg']], dtype=object)

        B['A_bf'] = scipy.sparse.csc_matrix([[1.0],
                                             [0.0],
                                             [0.0],
                                             [0.0]])

        B['PRO_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])

        B['A_ff'] = scipy.sparse.csc_matrix([[11.0]])

        # WITH THREE STRESSORS

        B['STR'] = np.array([['stress01', 1614, 'kg'],
                             ['stress02', 1615, 'kg'],
                             ['stress03', 1616, 'kg']], dtype=object)

        B['F_f'] = scipy.sparse.csc_matrix([[ 0.0],
                                            [ 0.2],
                                            [ 0.0]])

        B['y_f'] = scipy.sparse.csc_matrix([[2.0]])

        b.extract_background_from_matdict(self.matdict)
        b.extract_foreground_from_matdict(B)

        a.append_to_foreground(b)

        y_f = pd.DataFrame({0: {10005: 1.0, 10002: 0.0, 10: 0.0}})
        assert_frames_equivalent(a.y_f, y_f)


    def test_import_and_export_matdict_keys_roundtrip(self):

        a = ArdaInventory.ArdaInventory()
        a.extract_background_from_matdict(self.matdict)
        a.extract_foreground_from_matdict(self.matdict)
        matdict = a.export_system_to_matdict(False)

        assert(matdict.keys() == self.matdict.keys())

    def test_extract_io_background_from_pymrio(self):

        mrio = pymrio.load_test()
        mrio.calc_all()
        a = ArdaInventory.ArdaInventory()
        a.extract_background_from_matdict(self.matdict)
        a.extract_io_background_from_pymrio(mrio)

        # assertion1
        assert(np.all(a.A_io.values == mrio.A.values))

        # assertion2: stest extensions preserved
        self.assertAlmostEqual(a.F_io.values.sum(), mrio.emissions.S.values.sum() +
                                      mrio.factor_inputs.S.values.sum())

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


    def test_delete_processes_foreground(self):
        a = ArdaInventory.ArdaInventory(1)
        a.extract_foreground_from_matdict(self.matdict)
        a.delete_processes_foreground([10005])

        # A 2X2 FOREGROUND

        B = {}
        B['PRO_f'] = np.array([ ['Batt Packing', 10002, 'kg']],
                                          dtype=object)

        B['A_bf'] = scipy.sparse.csc_matrix([[1],
                                             [0],
                                             [0],
                                             [0]])

        B['PRO_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])

        B['A_ff'] = scipy.sparse.csc_matrix([[11]])

        # WITH THREE STRESSORS

        B['STR'] = np.array([['stress01', 1614, 'kg'],
                                        ['stress02', 1615, 'kg'],
                                        ['stress03', 1616, 'kg']], dtype=object)

        B['F_f'] = scipy.sparse.csc_matrix([[ 0.0],
                                            [ 0.2],
                                            [ 0.0]])

        B['y_f'] = scipy.sparse.csc_matrix([[0.0]])

        b = ArdaInventory.ArdaInventory(1)
        b.extract_background_from_matdict(self.matdict)
        b.extract_foreground_from_matdict(B)
        pdt.assert_frame_equal(a.A_ff, b.A_ff)
        pdt.assert_frame_equal(a.F_f, b.F_f)
        pdt.assert_frame_equal(a.y_f, b.y_f)

        assert(np.all(a.A_bf.values == b.A_bf.values))
        assert(np.all(a.PRO_f == b.PRO_f))

    def test_append_to_foreground_w_ValueError(self):
        a = ArdaInventory.ArdaInventory()
        b = ArdaInventory.ArdaInventory()
        a.extract_foreground_from_matdict(self.matdict)
        b.extract_foreground_from_matdict(self.matdict)

        with self.assertRaises(ValueError):
            a.append_to_foreground(b)

    def test_append_to_foreground_with_final_demand(self):
        a = ArdaInventory.ArdaInventory(1)
        b = ArdaInventory.ArdaInventory(1)
        a.extract_background_from_matdict(self.matdict)
        b.extract_background_from_matdict(self.matdict)

        B = {}
        B['PRO_f'] = np.array([ ['foo', 10, 'kg']],
                                          dtype=object)

        B['A_bf'] = scipy.sparse.csc_matrix([[1.0],
                                             [0.0],
                                             [0.0],
                                             [0.0]])

        B['PRO_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])

        B['A_ff'] = scipy.sparse.csc_matrix([[11]])

        # WITH THREE STRESSORS

        B['STR'] = np.array([['stress01', 1614, 'kg'],
                             ['stress02', 1615, 'kg'],
                             ['stress03', 1616, 'kg']], dtype=object)

        B['F_f'] = scipy.sparse.csc_matrix([[ 0.0],
                                            [ 0.2],
                                            [ 0.0]])

        B['y_f'] = scipy.sparse.csc_matrix([[2.0]])

        a.extract_foreground_from_matdict(self.matdict)
        b.extract_foreground_from_matdict(B)

        a.append_to_foreground(b, final_demand=True)


        A_ff = pd.DataFrame({10005: {10005: 0.0, 10002: 10.0, 10: 0.0},
                             10: {10005: 0.0, 10002: 0.0, 10: 11.0},
                             10002: {10005: 1.0, 10002: 11.0, 10: 0.0}})
        assert_frames_equivalent(a.A_ff, A_ff)

        A_bf = pd.DataFrame({10005: {1: 0.0, 2: 1.0, 3: 1.0, 4: 0.0},
                             10002: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0},
                             10:    {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}})

        assert_frames_equivalent(a.A_bf, A_bf)

        F_f = pd.DataFrame(
                {10005: {1616: 0.0, 1614: 0.29999999999999999, 1615: 0.10000000000000001},
                 10002: {1616: 0.0, 1614: 0.0, 1615: 0.20000000000000001},
                 10: {1616: 0.0, 1614: 0.0, 1615: 0.20000000000000001}})

        assert_frames_equivalent(a.F_f, F_f)

        y_f = pd.DataFrame({0: {10005: 1.0, 10002: 0.0, 10: 2.0}})

        assert_frames_equivalent(a.y_f, y_f)

        # check that order is right
        assert(np.all(a.PRO_f.ix[:,a._arda_default_labels] == a.A_ff.index.values))

    def test_append_to_foreground_w_ValueError_w_multiindex(self):
        index = [0, 1 , -1]
        a = ArdaInventory.ArdaInventory(index)
        b = ArdaInventory.ArdaInventory(index)
        a.extract_foreground_from_matdict(self.matdict)
        b.extract_foreground_from_matdict(self.matdict)

        with self.assertRaises(ValueError):
            a.append_to_foreground(b)

    def test_append_to_foreground_multiindex(self):
        a = ArdaInventory.ArdaInventory([0,1,-1])
        b = ArdaInventory.ArdaInventory([0, 1, -1])
        a.extract_foreground_from_matdict(self.matdict)

        B = {}
        B['PRO_f'] = np.array([ ['foo', 10, 'kg']],
                                          dtype=object)

        B['PRO_gen'] = np.array([['back01', 1, 'kg'],
                                            ['back02', 2, 'kg'],
                                            ['back03', 3, 'MJ'],
                                            ['back04', 4, 'MJ']], dtype=object)

        B['A_bf'] = scipy.sparse.csc_matrix([[1.0],
                                             [0.0],
                                             [0.0],
                                             [0.0]])

        B['PRO_header'] = np.array([['FULL NAME', 'MATRIXID','UNIT']])

        B['A_ff'] = scipy.sparse.csc_matrix([[11.0]])

        # WITH THREE STRESSORS

        B['STR'] = np.array([['stress01', 1614, 'kg'],
                             ['stress02', 1615, 'kg'],
                             ['stress03', 1616, 'kg']], dtype=object)

        B['F_f'] = scipy.sparse.csc_matrix([[ 0.0],
                                            [ 0.2],
                                            [ 0.0]])

        B['y_f'] = scipy.sparse.csc_matrix([[0]])

        b.extract_background_from_matdict(self.matdict)
        b.extract_foreground_from_matdict(B)

        a.append_to_foreground(b)


        A_bf = pd.DataFrame( {('foo', 10, 'kg'): {('back01', 1, 'kg'): 1.0,
                                                  ('back03', 3, 'MJ'): 0.0,
                                                  ('back04', 4, 'MJ'): 0.0,
                                                  ('back02', 2, 'kg'): 0.0},
                              ('Batt Packing', 10002, 'kg'):
                                                 {('back01', 1, 'kg'): 1.0,
                                                  ('back03', 3, 'MJ'): 0.0,
                                                  ('back04', 4, 'MJ'): 0.0,
                                                  ('back02', 2, 'kg'): 0.0},
                              ('s+orm', 10005, 'kg'): {
                                                  ('back01', 1, 'kg'): 0.0,
                                                  ('back03', 3, 'MJ'): 1.0,
                                                  ('back04', 4, 'MJ'): 0.0,
                                                  ('back02', 2, 'kg'): 1.0}
                              })

        assert_frames_equivalent(a.A_bf, A_bf)

    def test_change_process_ids(self):
        a = ArdaInventory.ArdaInventory([1])
        a.extract_foreground_from_matdict(self.matdict)

        a.increase_foreground_process_ids(70000)

        A_ff = pd.DataFrame({80005: {80005: 0, 80002: 10},
                             80002: {80005: 1, 80002: 11}})
        assert_frames_equivalent(a.A_ff, A_ff)


        PRO_f =  np.array([['s+orm', 80005, 'kg'],
                           ['Batt Packing', 80002, 'kg']], dtype=object)
        assert(np.all(a.PRO_f == PRO_f))

    def test_properties_singleindex(self):
        mrio = pymrio.load_test()
        mrio.calc_all()
        a = ArdaInventory.ArdaInventory([1])
        a.extract_background_from_matdict(self.matdict)
        a.extract_foreground_from_matdict(self.matdict)
        a.extract_io_background_from_pymrio(mrio)

        a.A
        a.F
        a.PRO
        a.STR_all
        a.C_all

    def test_properties_multiindex(self):
        mrio = pymrio.load_test()
        mrio.calc_all()
        a = ArdaInventory.ArdaInventory([0,1])
        a.extract_background_from_matdict(self.matdict)
        a.extract_foreground_from_matdict(self.matdict)
        a.extract_io_background_from_pymrio(mrio)

        a.A
        a.F
        a.PRO
        a.STR_all
        a.C_all

#=========================================================
def assert_frames_equivalent(df1, df2, **kwds):
    pdt.assert_frame_equal(df1.sort_index().sort(axis=1),
                           df2.sort_index().sort(axis=1),
                           **kwds)

if __name__ == '__main__':
        unittest.main()
