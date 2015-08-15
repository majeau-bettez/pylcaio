import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse
import sys

sys.path.append('/home/bill/software/Python/Modules/')
import matlab_tools as mlt
import matrix_view as mtv

class ArdaInventory(object):
    """ A common data structure for an Arda template"""


    def __init__(self, index_columns=[0, 1, -1]):
    
        # Main matrices, as Pandas Dataframes ()
        self.A_ff = None
        self.A_bf = None
        self.A_bb = None
        self.A_fb = None
    
        self.F_f = None
        self.F_b = None
        self.C = None
        self.y_f = None
        self.y_b = None
    
        # Labels as numpy arrays
        self.PRO_f = None
        self.PRO_b = None
        self.STR = None
        self.IMP = None
        self.PRO_header = None
        
        # Index Lists
        self.index_str = None
        self.index_pro_b = None
        self.index_pro_f = None
        self.index_imp = None

        self._arda_default_labels = index_columns



    def extract_labels_from_matdict(self, matdict):

        try:
            self.STR = mlt.mine_nested_array(matdict['STR'])
            self.index_str = self.STR[:,self._arda_default_labels].T.tolist()
        except:
            pass

        try:
            self.PRO_b = mlt.mine_nested_array(matdict['PRO_gen'])
            self.index_pro_b = self.PRO_b[:, self._arda_default_labels].T.tolist()
        except:
            pass

        try:
            self.IMP = mlt.mine_nested_array(matdict['IMP'])
            self.index_imp = self.IMP[:, self._arda_default_labels].T.tolist()
        except:
            pass

        try:
            self.PRO_f = mlt.mine_nested_array(matdict['PRO_f'])
            self.index_pro_f = self.PRO_f[:, self._arda_default_labels].T.tolist()
        except:
            pass

        try:
            self.PRO_header = mlt.mine_nested_array(matdict['PRO_header'])
        except:
            pass

    def extract_background_from_matdict(self, matdict):
        
        self.extract_labels_from_matdict(matdict)
        
        self.F_b = pd.DataFrame(data=matdict['F_gen'].toarray(),
                                index=self.index_str,
                                columns=self.index_pro_b)
        self.A_bb = pd.DataFrame(data=matdict['A_gen'].toarray(),
                                 index=self.index_pro_b,
                                 columns=self.index_pro_b)
        self.C = pd.DataFrame(data=matdict['C'].toarray(),
                              index=self.index_imp,
                              columns=self.index_str)
        try:
            self.y_b = pd.DataFrame(data=matdict['y_gen'].toarray(),
                                    index=self.index_pro_b)
        except:
            pass
        
    def extract_foreground_from_matdict(self, matdict):
        
        self.extract_labels_from_matdict(matdict)
        
        
        self.A_ff = pd.DataFrame(data=matdict['A_ff'].toarray(),
                                 index=self.index_pro_f,
                                 columns=self.index_pro_f)
        self.A_bf = pd.DataFrame(data=matdict['A_bf'].toarray(),
                                 index=self.index_pro_b,
                                 columns=self.index_pro_f)
        
        self.F_f = pd.DataFrame(data=matdict['F_f'].toarray(),
                                index=self.index_str,
                                columns=self.index_pro_f)
        
        try:
            self.y_f = pd.DataFrame(data=matdict['y_f'].toarray(),
                                    index=self.index_pro_f)
        except:
            raise Warning('No final demand found')
        
    def match_foreground_to_background(self):
        
        F_f_new = self.F_f.reindex_axis(self.F_b.index, axis=0).fillna(0)
        
        if F_f_new.sum().sum() != self.F_f.sum().sum():
            raise ValueError('Some of the emissions are not conserved during'
                    ' the re-indexing! Will not re-index F_f')
        else:
            self.F_f = F_f_new
            
        A_bf_new = self.A_bf.reindex_axis(self.A_bb.index, axis=0).fillna(0)
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
                    'PRO_f': self.PRO_f,
                    'PRO_gen': self.PRO_b,
                    'STR': self.PRO_b
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
                    'PRO_gen': self.PRO_b,
                    'STR': self.PRO_b,
                    'IMP': self.IMP
                   }
        matdict.update(matdict_fore)

        return matdict
        
    

