import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse
import sys
import IPython

sys.path.append('/home/bill/software/Python/Modules/')
import matlab_tools as mlt
import matrix_view as mtv

class ArdaInventory(object):
    """ A common data structure for an Arda template"""


    def __init__(self, index_columns=[1]):

        # extended Labels
        self.PRO_f = pd.DataFrame()
        self.PRO_b = pd.DataFrame()
        self.STR = pd.DataFrame()
        self.IMP = pd.DataFrame()
        self.PRO_io = pd.DataFrame()
        self.STR_io = pd.DataFrame()
        self.IMP_io = pd.DataFrame()

        # label column headers
        self.PRO_header = np.array([])

        # Main matrices, as Pandas Dataframes ()
        self.A_ff = pd.DataFrame()
        self.A_bf = pd.DataFrame()
        self.A_bb = pd.DataFrame()

        self.F_f = pd.DataFrame()
        self.F_b = pd.DataFrame()
        self.C = pd.DataFrame()
        self.y_f = pd.DataFrame()
        self.y_b = pd.DataFrame()

        #self.A_fb = pd.DataFrame()


        self._arda_default_labels = index_columns
        self._ardaId_column = 1

        self.A_io = pd.DataFrame()
        self.A_io_f = pd.DataFrame()
        self.F_io = pd.DataFrame()
        self.F_io_f = pd.DataFrame()

        self.io_material_sectors=np.array([])

    @property
    def A(self):
        return pd.concat([
                 self.A_ff,
                 pd.concat([self.A_bf, self.A_bb], axis =1),
                 pd.concat([self.A_io_f, self.A_io], axis=1)], axis=0)

    @property
    def F(self):
        return pd.concat(
                [pd.concat([self.F_f, self.F_b], axis=1),
                 pd.concat([self.F_io_f, self.F_io], axis=1)], axis=0)

    def extract_labels_from_matdict(self, matdict):

        try:
            STR = mlt.mine_nested_array(matdict['STR'])
            self.STR = pd.DataFrame(
                            data=STR,
                            index=STR[:,self._arda_default_labels].T.tolist())
        except:
            pass

        try:
            PRO_b = mlt.mine_nested_array(matdict['PRO_gen'])
            self.PRO_b = pd.DataFrame(
                    data=PRO_b,
                    index=PRO_b[:, self._arda_default_labels].T.tolist()
                    )
        except:
            pass

        try:
            IMP = mlt.mine_nested_array(matdict['IMP'])
            self.IMP = pd.DataFrame(
                    data=IMP,
                    index=IMP[:, self._arda_default_labels].T.tolist()
                    )
        except:
            pass

        try:
            PRO_f = mlt.mine_nested_array(matdict['PRO_f'])
            self.PRO_f = pd.DataFrame(
                    data = PRO_f,
                    index = PRO_f[:,self._arda_default_labels].T.tolist()
                    )
        except:
            pass

        try:
            self.PRO_header = mlt.mine_nested_array(matdict['PRO_header'])
        except:
            pass

    def extract_background_from_matdict(self, matdict):
        
        self.extract_labels_from_matdict(matdict)
        
        self.F_b = pd.DataFrame(data=matdict['F_gen'].toarray(),
                                index=self.STR.index,
                                columns=self.PRO_b.index)
        self.A_bb = pd.DataFrame(data=matdict['A_gen'].toarray(),
                                 index=self.PRO_b.index,
                                 columns=self.PRO_b.index)
        self.C = pd.DataFrame(data=matdict['C'].toarray(),
                              index=self.IMP.index,
                              columns=self.STR.index)
        try:
            self.y_b = pd.DataFrame(data=matdict['y_gen'].toarray(),
                                    index=self.PRO_b.index)
        except:
            pass
        
    def extract_foreground_from_matdict(self, matdict):
        
        self.extract_labels_from_matdict(matdict)
        
        self.A_ff = pd.DataFrame(data=matdict['A_ff'].toarray(),
                                 index=self.PRO_f.index,
                                 columns=self.PRO_f.index)
        self.A_bf = pd.DataFrame(data=matdict['A_bf'].toarray(),
                                 index=self.PRO_b.index,
                                 columns=self.PRO_f.index)
        
        self.F_f = pd.DataFrame(data=matdict['F_f'].toarray(),
                                index=self.STR.index,
                                columns=self.PRO_f.index)

        try:
            self.y_f = pd.DataFrame(data=matdict['y_f'].toarray(),
                                    index=self.PRO_f.index)
        except:
            raise Warning('No final demand found')

    def extract_io_background(self,mrio):
        self.A_io = mrio.A.copy()
        self.F_io = pd.concat([mrio.emissions.S, mrio.factor_inputs.S])

        self.A_io_f = pd.DataFrame(index=self.A_io.index,
                                   columns=self.A_ff.columns).fillna(0.0)

        self.F_io_f = pd.DataFrame(index=self.F_io.index,
                                   columns=self.A_ff.columns).fillna(0.0)

    def match_foreground_to_background(self):

        F_f_new = self.F_f.reindex_axis(self.F_b.index, axis=0).fillna(0.0)

        if F_f_new.sum().sum() != self.F_f.sum().sum():
            raise ValueError('Some of the emissions are not conserved during'
                    ' the re-indexing! Will not re-index F_f')
        else:
            self.F_f = F_f_new

        A_bf_new = self.A_bf.reindex_axis(self.A_bb.index, axis=0).fillna(0.0)
        if A_bf_new.sum().sum() != self.A_bf.sum().sum():
            raise ValueError('Some of the product-flows are not conserved'
                    ' during the re-indexing! Will not re-index A_bf')
        else:
            self.A_bf = A_bf_new
            
    def export_foreground_to_matdict(self):
        matdict = {
                    'A_ff': scipy.sparse.csc_matrix(self.A_ff.values),
                    'A_bf': scipy.sparse.csc_matrix(self.A_bf.values),
                    'F_f': scipy.sparse.csc_matrix(self.F_f.values),
                    'y_f': scipy.sparse.csc_matrix(self.y_f.values),
                    'PRO_f': self.PRO_f.values,
                    'PRO_gen': self.PRO_b.values,
                    'STR': self.STR.values
                   }

        try:
            matdict['PRO_header'] = self.PRO_header
        except:
            pass

        return matdict

    def export_system_to_matdict(self):

        matdict_fore = self.export_foreground_to_matdict()
        matdict = {
                    'A_gen': scipy.sparse.csc_matrix(self.A_bb.values),
                    'F_gen': scipy.sparse.csc_matrix(self.F_f.values),
                    'C': scipy.sparse.csc_matrix(self.C.values),
                    'y_gen': scipy.sparse.csc_matrix(self.y_f.values),
                    'PRO_gen': self.PRO_b.values,
                    'STR': self.STR.values,
                    'IMP': self.IMP
                   }
        matdict.update(matdict_fore)

        return matdict
        
    def delete_processes_foreground(self, id_begone=[]):

            #bo_begone = np.zeros(self.PRO_f.shape[0], dtype=bool)
            #for i in id_begone:
            #    bo_begone += self.PRO_f.ix[:, self._ardaId_column] == i

            self.PRO_f = self.PRO_f.drop(id_begone, 0)
            self.A_ff = self.A_ff.drop(id_begone, 0).drop(id_begone, 1)
            self.A_bf = self.A_bf.drop(id_begone, 1)
            self.F_f = self.F_f.drop(id_begone, 1)
            self.y_f = self.y_f.drop(id_begone, 0)

    def append_to_foreground(self, other, final_demand=False):


        # check if no duplicate index in dataframes
        if len(self.A_ff.index - other.A_ff.index) < len(self.A_ff.index):
            raise ValueError("The two inventories share common foreground"
                             " labels. I will not combine them.")

        # double check with label
        diff = (self.PRO_f.ix[:, self._ardaId_column] -
                other.PRO_f.ix[:, other._ardaId_column])
        if len(diff) < self.PRO_f.shape[0]:
            raise ValueError("The two inventories share common ardaIds"
                             " labels. I will not combine them.")

        # concatenate labels

        # for the Ids, make sure that they all have the same type. Software
        # like matlab complain when different types are mixed (e.g., int64,
        # uint16, etc.)
        the_type = type(other.PRO_f.iloc[0, self._ardaId_column])
        self.PRO_f = pd.concat([self.PRO_f, other.PRO_f], axis=0)
        self.PRO_f[self._ardaId_column] = self.PRO_f[self._ardaId_column].astype(the_type)


        def concat_keep_order(frame_list, index, axis=0, order_axis=[0]):
            c = pd.concat(frame_list, axis).fillna(0.0)
            for i in order_axis:
                c = c.reindex_axis(index, axis=i)
            return c

        # concatenate data
        self.A_ff = concat_keep_order([self.A_ff, other.A_ff],
                                      self.PRO_f.index,
                                      order_axis=[0,1])


        self.A_bf = concat_keep_order([self.A_bf, other.A_bf],
                                      self.PRO_f.index,
                                      axis=1,
                                      order_axis=[1])

        self.F_f = concat_keep_order([self.F_f, other.F_f],
                                     self.PRO_f.index,
                                     axis=1,
                                     order_axis=[1])

        if final_demand:
            self.y_f = concat_keep_order([self.y_f, other.y_f],
                                         self.PRO_f.index,
                                         axis=0,
                                         order_axis=[0])
        else:
            self.y_f = self.y_f.reindex(self.A_ff.index).fillna(0.0)


    def increase_foreground_process_ids(self, shift=0, index_col=0):

        self.PRO_f.ix[:, self._ardaId_column] += shift

        if isinstance(self.A_ff.index, pd.core.index.MultiIndex):

            index = np.array([list(r) for r in self.A_ff.index])
            index[:, index_col] += shift
            index = pd.MultiIndex.from_array(index.T)

            self.PRO_f.index = index
            self.A_ff.index = index
            self.A_ff.columns = index
            self.y_f.index = index
            self.A_bf.columns = index
            self.F_f.columns = index
        else:
            self.PRO_f.index += shift
            self.A_ff.index += shift
            self.A_ff.columns += shift
            self.y_f.index += shift
            self.A_bf.columns += shift
            self.F_f.columns += shift
            

    def hybridize_process(self, fore_id, region, sector, price,
            full_material=True, full_emissions=True,
            doublecounted_sectors = None):

        self.A_io_f.ix[:,fore_id] = self.A_io[region, sector]*price

        if full_material:
            all_sectors = self.A_io_f.index.get_level_values(1)
            bo = all_sectors.isin(self.io_material_sectors)
            self.A_io_f.ix[bo,fore_id] = 0.0

        if not full_emissions:
            self.F_io_f.ix[:, fore_id] =self.F_io[region, sector]*price

        if doublecounted_sectors is not None:
            for i in doublecounted_sectors:
                self.A_io_f.ix[i, fore_id] = 0.0


