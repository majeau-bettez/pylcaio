import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse
import sys
import IPython
import copy

sys.path.append('/home/bill/software/Python/Modules/')
import matlab_tools as mlt
import matrix_view as mtv

class ArdaInventoryHybridizer(object):
    """Object to handle an LCA inventory and hybridize it with an EEIO table
    """


    def __init__(self, index_columns=[1]):

        # extended Labels
        self.PRO_f = pd.DataFrame()
        self.PRO_b = pd.DataFrame()
        self.STR = pd.DataFrame()
        self.IMP = pd.DataFrame()
        self.PRO_io = pd.DataFrame()
        self.STR_io = pd.DataFrame()
        self.IMP_io = pd.DataFrame()

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
        self.C_io = pd.DataFrame()

        self.io_material_sectors=np.array([])

        self.hyb = pd.DataFrame(columns=['process_index',
                                         'io_index',
                                         'price_per_fu'])


# PROPERTIES
    @property
    def PRO(self):
        """ Process/sector labels for whole system """
        pro =  pd.concat([self.PRO_f, self.PRO_b, self.PRO_io], axis=0)
        return reorder_cols(pro.fillna(''))

    @property
    def STR_all(self):
        str_all = pd.concat([self.STR, self.STR_io], axis=0)
        return reorder_cols(str_all.fillna(''))

    @property
    def IMP_all(self):
        imp = pd.concat([self.IMP, self.IMP_io], axis=0)
        return reorder_cols(imp.fillna(''))

    @property
    def A(self):
        a = pd.concat([     i2s(self.A_ff),
                 pd.concat([i2s(self.A_bf), i2s(self.A_bb)], axis=1),
                 pd.concat([i2s(self.A_io_f), i2s(self.A_io)], axis=1)], axis=0
                 )
        return a.reindex_axis(self.PRO.index, 0).reindex_axis(self.PRO.index,1
                                                              ).fillna(0.0)
    @property
    def F(self):
        f= pd.concat([pd.concat([i2s(self.F_f), i2s(self.F_b)], axis=1),
                      pd.concat([i2s(self.F_io_f), i2s(self.F_io)], axis=1)],
                     axis=0)
        return f.reindex_axis(self.STR_all.index, 0).reindex_axis(self.PRO.index,1
                ).fillna(0.0)

    @property
    def C_all(self):
        return concat_keep_order([i2s(self.C), i2s(self.C_io)],
                                 i2s(self.STR_all).index,
                                 order_axis=[1])

    @property
    def y(self):
        y_pro =  pd.concat([self.y_f, self.y_b], axis=0)
        return y_pro.reindex_axis(self.PRO.index, axis=0).fillna(0.0)

#=============================================================================
# METHODS
#=============================================================================
    def extract_labels_from_matdict(self, matdict, overrule):

        if  (overrule or len(self.STR) == 0) and 'STR' in matdict:
            STR = mlt.mine_nested_array(matdict['STR'])
            self.STR = pd.DataFrame(
                            data=STR,
                            index=STR[:, self._arda_default_labels].T.tolist()
                            )
            try:
                STR_header = mlt.mine_nested_array(matdict['STR_header'])
                self.STR.columns = extract_header(STR_header)
            except:
                pass


        if  (overrule or len(self.PRO_b) == 0) and 'PRO_gen' in matdict:
            PRO_b = mlt.mine_nested_array(matdict['PRO_gen'])
            PRO_header = mlt.mine_nested_array(matdict['PRO_header'])
            self.PRO_b = pd.DataFrame(
                    data=PRO_b,
                    columns = extract_header(PRO_header),
                    index=PRO_b[:, self._arda_default_labels].T.tolist()
                    )

        if  (overrule or len(self.IMP) == 0) and 'IMP' in matdict:
            IMP = mlt.mine_nested_array(matdict['IMP'])
            IMP_header = mlt.mine_nested_array(matdict['IMP_header'])
            self.IMP = pd.DataFrame(
                    data=IMP,
                    columns=extract_header(IMP_header),
                    index=IMP[:, self._arda_default_labels].T.tolist()
                    )

        if  (overrule or len(self.PRO_f) == 0) and 'PRO_f' in matdict:
            PRO_f = mlt.mine_nested_array(matdict['PRO_f'])
            try:
                PRO_header = extract_header(
                        mlt.mine_nested_array(matdict['PRO_header']))
            except:
                if len(self.PRO_b.columns) == PRO_f.shape[1]:
                    PRO_header = self.PRO_b.columns
                else:
                    raise Exception("Cannot read PRO_header")
            self.PRO_f = pd.DataFrame(
                    data = PRO_f,
                    index = PRO_f[:, self._arda_default_labels].T.tolist(),
                    columns = PRO_header
                    )

    def extract_background_from_matdict(self, matdict, overrule=True):

        self.extract_labels_from_matdict(matdict, overrule)

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

    def extract_foreground_from_matdict(self, matdict, overrule=True):

        self.extract_labels_from_matdict(matdict, overrule)

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

    def reconcile_ids(self, io_label, arda_label, header):

        # calculate smallest id not conflicting with that of arda_label
        a = np.max(arda_label.iloc[:,self._ardaId_column])
        order_magnitude = int(np.math.floor(np.math.log10(abs(a))))
        min_id = np.around(a, -order_magnitude) + 10**order_magnitude + 1
        new_ids = np.array(
                [i for i in range(min_id, min_id  + io_label.shape[0])]
                ).reshape((io_label.shape[0], 1))

        new_iolabels = np.hstack((io_label[:, :self._ardaId_column],
                                  new_ids,
                                  io_label[:, self._ardaId_column:]))
        new_header = (header[: self._ardaId_column]
                             + [arda_label.columns[self._ardaId_column]]
                             + header[self._ardaId_column:])

        return new_iolabels, new_header

    def extract_io_background_from_pymrio(self, mrio, pro_name_cols=None,
            str_name_cols=None,): 




        # Clean up MRIO
        mrio.reset_all_to_coefficients()
        if mrio.unit is None:
            mrio.unit = pd.DataFrame(index=mrio.A.index, columns=['UNIT'])
            mrio.unit.loc[:,'UNIT'] = 'M.EUR.'


        self.A_io = mrio.A.copy()


        # get "process labels", add units as last column
        PRO_io = np.hstack((
                    np.array([list(r) for r in self.A_io.index]),
                    mrio.unit.values))
        PRO_header = [x.upper() for x in mrio.A.index.names]
        PRO_header = PRO_header + ['UNIT']

        # define a "fullname" column as the first column of the process labels
        if pro_name_cols is not None:
            PRO_io, PRO_header = generate_fullname(PRO_io,
                                                   PRO_header,
                                                   pro_name_cols)
#            fullname = np.empty((PRO_io.shape[0], 1), dtype='object')
#            for i in range(PRO_io.shape[0]):
#                fullname[i] = '/ '.join(PRO_io[i, pro_name_cols])
#            PRO_io = np.hstack((fullname, PRO_io))
#            PRO_header = ['FULL NAME'] + PRO_header

        # define some numeric ID for each process, put in second column
        PRO_io, PRO_header= self.reconcile_ids(PRO_io, self.PRO, PRO_header)

        self.PRO_io = pd.DataFrame(PRO_io,
                                   #index=PRO_io[:,self._arda_default_labels].T.tolist(),
                                   index=self.A_io.index,
                                   columns = PRO_header)

        # Check if we have a mix of single-index and multi-index dataframes in
        # the MRIO extensions
        bo_singleIndex = False
        max_names = 1
        for i in mrio.get_extensions(True):
            names_length = len(i.S.index.names)
            if names_length > max_names:
                max_names = names_length
                max_headers = i.S.index.names
            elif names_length == 1:
                bo_singleIndex=True

        # Combine all extensions as one
        units_str=pd.DataFrame([])
        for i in mrio.get_extensions(True):
            # if necessary, turn single-index dataframes in multiIndex ones
#            if (max_names > 1 and bo_singleIndex and not isinstance(i.S.index, pd.core.index.MultiIndex)):
# TODO: NEEDS TESTING next line
            i.S.index = augment_index(i.S.index, max_names)
            self.F_io = pd.concat([self.F_io, i.S])
            units_str = pd.concat([units_str, i.unit])
            self.F_io.index.names = max_headers



        # get STR labels and units (as last column)
        STR_io = np.hstack((np.array([list(r) for r in self.F_io.index]),
                            units_str.values))

        # get STR header
        STR_header = [x.upper() for x in self.F_io.index.names]
        STR_header = STR_header + ['UNIT']


        # define a fullname for each stressor
        if str_name_cols is not None:
            STR_io, STR_header = generate_fullname(STR_io,
                                                   STR_header,
                                                   str_name_cols)

        # define some numeric ID for each stressor, put in second column
        STR_io, STR_header= self.reconcile_ids(STR_io, self.STR, STR_header)

        self.STR_io = pd.DataFrame(
                STR_io,
                #index=STR_io[:, self._arda_default_labels].T.tolist(),
                index=self.F_io.index,
                columns=STR_header
                )

        self.A_io_f = pd.DataFrame(index=self.A_io.index,
                                   columns=self.A_ff.columns).fillna(0.0)

        self.F_io_f = pd.DataFrame(index=self.F_io.index,
                                   columns=self.A_ff.columns).fillna(0.0)

    def extract_exiobase2_characterisation_factors(self,
            char_filename='characterisation_CREEA_version2.2.0.xlsx',
            xlschar_param=None, name_cols=[0]):

        if xlschar_param is None:
            xlschar_param = {
                    'Q_emission':
                        # INDEX HEADERS    ,  COL HEADERS , ROWS_DROP, COL_DROP
                        (['impact', 'unit'], ['stressor','comp'],[2], [0,2]),
                    'Q_factorinputs':
                        (['impact', 'unit'], ['stressor'], [1], None),
                    'Q_resources':
                        (['impact', 'unit'], ['stressor'], [1,2], None),
                    'Q_materials':
                        (['impact', 'unit'], ['stressor'], [1] , None)}

            # find widest index and columns
            max_cols=0
            max_index=0
            for key in xlschar_param.keys():
                index_width = len(xlschar_param[key][0])
                if index_width > max_index:
                    max_index = copy.deepcopy(index_width)
                col_width = len(xlschar_param[key][1])
                if col_width > max_cols:
                    max_cols = copy.deepcopy(col_width)


        C = pd.DataFrame()
        for key in xlschar_param.keys():
            par = xlschar_param[key]
            c = extract_char(char_filename, key, par[0], par[1], par[2], par[3])
            c.index = augment_index(c.index, max_index)
            c.columns = augment_index(c.columns, max_cols)
            C = pd.concat([C, c], join='outer')

        # characterized stressors that would get lost in processing?
        diff = set(C.columns) - set(self.F_io.index)
        if len(diff):
            print("WARNING: some characterized stressors seem lost in matching.")

        # inventoried "stressors" that will not be characterized
        diff = set(self.F_io.index) - set(C.columns)
        if len(diff) == C.shape[0]:
            # mostly likely pre-characterised impacts that were part of the
            # stressor matrix: drop them
            self.F_io = self.F_io.drop(list(diff))
            self.STR_io = self.STR_io.drop(list(diff))
        elif len(diff) != 0:
            print("Warning, some inventoried stressors not characterized")
        else:
             pass

        self.C_io = C.reindex_axis(self.F_io.index, axis='columns').fillna(0)
        IMP_header = extract_header(self.C_io.index.names)
        IMP = np.array([list(i) for i in self.C_io.index.values.tolist()],
                       dtype=object)
        IMP, IMP_header = generate_fullname(IMP, IMP_header, name_cols)
        IMP, IMP_header = self.reconcile_ids(IMP, self.IMP, IMP_header)

        self.IMP_io = pd.DataFrame(index = self.C_io.index,
                                   columns = IMP_header,
                                   data = IMP)
        # todo: must have fullname, and numerical id
        #       UNIT, not unit
        #       NAME, not impact?

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
            
    def to_matfile(self, filename, foreground=True, background=True):

        csc = scipy.sparse.csc_matrix

        if foreground and not background:
            sio.savemat(filename, {
                    'A_ff':         scipy.sparse.csc_matrix(self.A_ff.values),
                    'A_bf':         scipy.sparse.csc_matrix(self.A_bf.values),
                    'F_f':          scipy.sparse.csc_matrix(self.F_f.values),
                    'y_f':          scipy.sparse.csc_matrix(self.y_f.values),
                    'PRO_f':        self.PRO_f.values,
                    'PRO_gen':      self.PRO_b.values,
                    'STR':          self.STR.values,
                    'PRO_header':   np.atleast_2d(self.PRO_f.columns.values),
                    'STR_header':   np.atleast_2d(self.STR.columns.values),
                    'IMP_header':   np.atleast_2d(self.IMP.columns.values)
                   })
        elif background and not foreground:
            print('Not tested')
            sio.savemat(filename, {
                    'A_gen': scipy.sparse.csc_matrix(self.A_bb.values),
                    'F_gen': scipy.sparse.csc_matrix(self.F_b.values),
                    'C': scipy.sparse.csc_matrix(self.C.values),
                    'y_gen': scipy.sparse.csc_matrix(self.y_b.values),
                    'PRO_gen': self.PRO_b.values,
                    'STR': self.STR.values,
                    'IMP': self.IMP.values,
                    'PRO_header': np.atleast_2d(self.PRO_b.columns.values),
                    'STR_header': np.atleast_2d(self.STR.columns.values),
                    'IMP_header': np.atleast_2d(self.IMP.columns.values)
                       })

        else:
            sio.savemat(filename, {
                    'A_gen':      csc(self.A.values),
                    'F_gen':      csc(self.F.values),
                    'C':          csc(self.C_all.values),
                    'y_gen':      csc(self.y.values),
                    'PRO_gen':    self.PRO.values,
                    'STR':        self.STR_all.values,
                    'IMP':        self.IMP_all.values,
                    'PRO_header': np.atleast_2d(self.PRO.columns.values),
                    'STR_header': np.atleast_2d(self.STR_all.columns.values),
                    'IMP_header': np.atleast_2d(self.IMP_all.columns.values)
                    })

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
        if len(self.A_ff.index.intersection(other.A_ff.index)) > 0:
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
        self.PRO_f.iloc[:,self._ardaId_column] = self.PRO_f.iloc[:,self._ardaId_column].astype(the_type)


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
            

    def hybridize_process(self,
                          process_index,
                          io_index,
                          price,
                          full_material=True,
                          full_emissions=True,
                          full_intrasector=True,
                          doublecounted_sectors=None,
                          sector_level_name='sector'):

        self.A_io_f.ix[:, process_index] = self.A_io.ix[:, io_index] * price

        all_sectors = self.A_io_f.index.get_level_values(sector_level_name)

        # get name of sector of interest
        bo = (self.A_io.index.to_series() == io_index).values
        sector = all_sectors[bo].tolist()[0]

        if full_material:
            bo = all_sectors.isin(self.io_material_sectors)
            self.A_io_f.ix[bo, process_index] = 0.0

        if full_intrasector:
            bo = all_sectors.isin([sector])
            self.A_io_f.ix[bo, process_index] = 0.0

        if not full_emissions:
            self.F_io_f.ix[:, process_index] = self.F_io[region, sector]*price

        if doublecounted_sectors is not None:
            for i in doublecounted_sectors:
                self.A_io_f.ix[i, process_index] = 0.0

    def hybridize_multiple_processes(self,
                                     full_material=True,
                                     full_emissions=True,
                                     full_intrasector=True,
                                     doublecounted_sectors=None,
                                     sector_level_name='sector'):

        for i, row in self.hyb.iterrows():
            self.hybridize_process(row.process_index,
                                   row.io_index,
                                   row.price_per_fu,
                                   full_material=full_material,
                                   full_emissions=full_emissions,
                                   full_intrasector=full_intrasector,
                                   doublecounted_sectors=doublecounted_sectors,
                                   sector_level_name=sector_level_name)

    def calc_lifecycle(self, stage):

        I = pd.DataFrame(np.eye(len(self.A)),
                         index=self.A.index,
                         columns=self.A.columns)
        x = pd.DataFrame(np.linalg.solve(I-self.A, self.y), index = self.A.index)

        if stage == 'production':
            return x

        e = self.F.dot(x)
        if stage == 'emissions':
            return e

        d = self.C_all.dot(e)
        if stage == 'impacts':
            return d


def concat_keep_order(frame_list, index, axis=0, order_axis=[0]):
    c = pd.concat(frame_list, axis).fillna(0.0)
    for i in order_axis:
        c = c.reindex_axis(index, axis=i)
    return c

def extract_header(header):
    """

    ARGS
    ----

    header:
        * Numpy array OR Pandas Dataframe that can be squeezed to 1D
        * OR 1D list of headers
   """ 


    try:
        # for numpy or pandas
        the_list = header.squeeze().tolist()
    except AttributeError:
        # for list
        the_list = header

    the_list = [x.upper() for x in the_list]
    the_list = [x.replace('ARDAID','MATRIXID') for x in the_list]
    the_list = [x.replace('FULL NAME','FULLNAME') for x in the_list]
    the_list = [x.replace('COMPARTMENT','COMP') for x in the_list]
    return the_list

def generate_fullname(label, header, name_cols):
    """ Prepend a "fullname" column before a label dataframe 

    """
    # define a "fullname" column as the first column of the process labels
    fullname = np.empty((label.shape[0], 1), dtype='object')
    for i in range(label.shape[0]):
        fullname[i] = '/ '.join(label[i, name_cols])
    label = np.hstack((fullname, label))
    header = ['FULLNAME'] + header
    return label, header

def reorder_cols(a):
    cols = a.columns - ['FULLNAME', 'MATRIXID','UNIT']
    sorted_cols = ['FULLNAME','MATRIXID'] + cols.tolist() + ['UNIT']
    return a.reindex_axis(sorted_cols, 1)

def i2s(a):
    a.index = a.index.to_series()
    a.columns = a.columns.to_series()
    return a

def extract_char(char_filename, sheet,
                 index_headers, col_headers, drop_rows, drop_cols):
    # width of columns and headers
    col_width=len(col_headers)
    index_width=len(index_headers)

    # read in whole sheet
    raw = pd.read_excel(char_filename,
                        sheet, index_col=None, header=None)
    # remove extraneous rows and columns
    if drop_rows is not None:
        raw = raw.drop(drop_rows, axis=0)
    if drop_cols is not None:
        raw = raw.drop(drop_cols, axis=1)

    # make a dataframe with selected indexes
    df = pd.DataFrame(
            data=raw.iloc[col_width:, index_width:].fillna(0).values,
            index=pd.MultiIndex.from_arrays(
                raw.iloc[col_width:, :index_width].fillna('').values.T,
                names=index_headers),
            columns=pd.MultiIndex.from_arrays(
                raw.iloc[:col_width, index_width:].fillna('').values,
                names=col_headers)
            )
    return df

def augment_index(index, width=None, headers=None):
    index = index.copy()
    tmp = []

    if width is None:
        width = len(headers)

    if len(index.names) == width:
        pass # all is fine
    elif len(index.names) == 1:
        for j in range(width):
            tmp.append(index.values)
        index = pd.MultiIndex.from_arrays(np.array(tmp))
    else:
        print("Warning, multi-index widening not implemented")

    if headers is not None:
        index.names=headers

    return index
