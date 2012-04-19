# -*- coding: utf-8 -*-

"""
GUI config.
@namespace ibs.ibs_gui
@details GUI config.
"""

from binary.conditionals_logistic import LogisticCondBinary
from binary.posterior_ml import PosteriorML
from binary.posterior_bvs import PosteriorBVS
from binary.product import ProductBinary
import Tkinter as tk
import copy
import csv
import numpy
import os
import tkFileDialog
import utils.configobj as configobj
import utils.pmw.Pmw as pmw


BUTTON = {'width':8}
STANDARD = {'padx':5, 'pady':5}

class GuiConfig(tk.Toplevel):

    def __init__(self, master, filename=''):
        '''
            Initialize configuration widget.
        '''
        tk.Toplevel.__init__(self, master)
        self.withdraw()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # read configuration
        self.filename = filename
        self.default = read_config(get_default_filename())
        self.myconfig = read_config(self.filename, default=self.default)
        self.title('Configuration - ' + self.config_name())

        # build interface
        self.balloon = pmw.Balloon(self)
        self.__init_widget()

        # center and show widget
        self.center_window()
        self.deiconify()

        self.protocol("WM_DELETE_WINDOW", self.close)

    def __init_widget(self):
        '''
            Build notebook.
        '''
        # create notebook from configuration
        self.notebook = pmw.NoteBook(self)
        self.notebook.grid(row=0, column=0, sticky='nswe', **STANDARD)

        self.entries = self.default.walk(lambda section, key : {})
        for section in self.default:
            page = self.notebook.add(GuiConfig.format_inline_comments(self.default, section)[0])
            group_number = 0
            for subsection in self.default[section]:
                group = pmw.Group(page, tag_text=GuiConfig.format_inline_comments(self.default[section], subsection)[0])
                group.grid(sticky='nswe', row=group_number / 2, column=group_number % 2, **STANDARD)
                self.__pack_entries(group.interior(),
                                    self.myconfig[section][subsection],
                                    self.entries[section][subsection])
                page.rowconfigure(group_number / 2, weight=1)
                page.columnconfigure(group_number % 2, weight=1)
                group_number += 1
        self.notebook.setnaturalsize()

        # add buttons
        button_frame = tk.Frame(self)
        self.buttons = {}
        button_frame.grid(row=1, column=0, sticky='nswe', **STANDARD)
        for text, command in [('Close', self.close),
                              ('Apply', self.write_config),
                              ('Reset', self.reset)]:
            self.buttons[text] = tk.Button(button_frame, text=text, command=command, **BUTTON)
            self.buttons[text].pack(side='right', **STANDARD)

    def __pack_entries(self, master, config, entries):
        '''
            Create entries in group.
        '''
        for key in config:
            args = {'label_text':key.replace('_', ' '), 'labelpos':'w', 'value': config[key]}
            inline = GuiConfig.format_inline_comments(config, key)

            # create entry
            if inline is None:
                # general field
                entry = pmw.EntryField(master, modifiedcommand=self.on_entry_change, **args)
            if isinstance(inline, str):
                # path field
                entry = PathEntryField(master, tk=self, default=self.default, dialog=inline,
                                       modifiedcommand=self.on_entry_change, **args)
            if isinstance(inline, list):
                # combo box
                value = args.pop('value')
                entry = pmw.ComboBox(master, scrolledlist_items=inline, listheight=len(inline) * 20,
                                     selectioncommand=self.on_entry_change, **args)
                try: entry.selectitem(value)
                except: pass
            entry.component('label').config(width=20, anchor=tk.W)
            entries[key] = entry

            # bind balloon tip
            self.balloon.bind(entry, GuiConfig.format_comments(config, key))

            # check operating system
            if (key.startswith('posix_') or key.startswith('nt_')):
                if not key.startswith(os.name): continue
            entry.pack(fill='x', **STANDARD)

    def on_entry_change(self, event=None):
        self.buttons['Apply'].config(text='*Apply')

    def config_name(self):
        if self.filename == get_default_filename(): return 'default'
        else: return os.path.basename(self.filename)

    def center_window(self):
        '''
            Center the configuration window.
        '''
        w = self.notebook.winfo_reqwidth()
        h = self.notebook.winfo_reqheight()
        ws = self.winfo_screenwidth()
        hs = self.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.geometry('+%d+%d' % (x, y))

    def write_config(self):
        '''
            Write configuration file.
        '''
        self.buttons['Apply'].config(text='Apply')

        # read values from fields
        self.myconfig.walk(GuiConfig.read_entries, entries=self.entries)

        if self.config_name() == 'default':
            self.myconfig.write()
            self.master.default = import_config(self.filename)
            self.master.refresh_run_file()
        else:
            write_config(self.myconfig, self.default)
            self.master.myconfig = import_config(self.filename)
        self.master.set_bar_color(self.master.myconfig['layout/color'])

    @staticmethod
    def read_entries(section, key, entries):
        '''
            Read the configuration from the entries.
        '''
        section[key] = entries[section.parent.name][section.name][key].get()

    def reset(self):
        '''
            Reset values.
        '''
        self.myconfig = read_config(self.filename, default=self.default)
        self.myconfig.walk(GuiConfig.write_entries, entries=self.entries)
        self.title('Configuration - ' + self.config_name())
        self.buttons['Apply'].config(text='Apply')

    @staticmethod
    def write_entries(section, key, entries):
        '''
            Write the configuration to the entries.
        '''
        entry = entries[section.parent.name][section.name][key]
        if isinstance(entry, pmw.ComboBox): entry.selectitem(section[key])
        else: entry.setvalue(section[key])

    def close(self):
        '''
            Close widet.
        '''
        self.master.config_window = None
        self.master.box_run_file.setvalue(self.config_name())
        self.destroy()

    @staticmethod
    def format_comments(section, key):
        '''
            Extract comments.
        '''
        return '\n'.join([line.replace('#', '').strip() for line in section.comments[key] if not line == ''])

    @staticmethod
    def format_inline_comments(section, key):
        '''
            Extract inline comments.
        '''
        inline = section.inline_comments[key]
        if inline is None: return
        if '$file' in inline: return '$file'
        if '$path' in inline: return '$path'
        return [option.strip() for option in inline.replace('#', '').split(',')]

    @staticmethod
    def remove_comments(myconfig):
        '''
            Remove all comments.
        '''
        for section in myconfig.sections:
            myconfig.comments[section] = []
            myconfig.inline_comments[section] = []
            for subsection in myconfig[section].sections:
                myconfig[section].comments[subsection] = []
                myconfig[section].inline_comments[subsection] = []
                for key in myconfig[section][subsection]:
                        myconfig[section][subsection].comments[key] = []
                        myconfig[section][subsection].inline_comments[key] = []


class PathEntryField(pmw.EntryField):

    def __init__(self, master, tk=None, default=None, dialog='$file', **args):
        '''
            Constructor.
        '''
        self.default = default
        self.tk = tk
        self.dialog = dialog
        pmw.EntryField.__init__(self, master, validate=self.validate, **args)
        self.component('entry').bind('<Double-Button-1>', self.dblclick)

    def validate(self, path):
        '''
            Validate the path.
        '''
        if os.path.isabs(path): return [pmw.PARTIAL, pmw.OK][os.path.exists(path)]
        if os.path.exists(os.path.join(get_root(), path)): return pmw.OK

        path = os.path.join(self.default['system']['path'][os.name + '_data'], path)
        if not os.path.isabs(path): path = os.path.join(get_root(), path)
        if os.path.exists(os.path.join(path)): return pmw.OK
        return pmw.PARTIAL

    def dblclick(self, text):
        '''
            Open file dialog.
        '''

        # starts directory dialog
        if self.dialog == '$path':
            args = {'title':'Choose a directory', 'parent':self.tk, 'initialdir':get_root()}
            dirname = tkFileDialog.askdirectory(**args)
            if len(dirname) > 0:
                self.setvalue(relpath(dirname, get_root()))

        # starts file dialog
        if self.dialog == '$file':
            args = {'title':'Choose a file', 'parent':self.tk}
            label = self.component('label').cget('text')

            # special csv file
            if label == 'csv file':
                args['filetypes'] = [('CSV-files', '*.csv')]
                args['initialdir'] = os.path.join(get_root(), self.default['system']['path'][os.name + '_data'])

            # open dialog
            filename = tkFileDialog.askopenfilename(**args)
            if filename == '': return

            if label == 'csv file':
                rel_path = relpath(filename, args['initialdir'])
                if not '..' in rel_path: filename = rel_path

            self.setvalue(filename)

def write_config(config, default=None):
    '''
        Write a configuration csv-file.
    '''
    # get default
    default_filename = get_default_filename()

    # save and return if config is default
    if config.filename == default_filename:
        config.write()
        return
    if default is None:
        default = configobj.ConfigObj(get_default_filename())

    myconfig = copy.deepcopy(config)
    for section in myconfig.itervalues():
        for subsection in section.itervalues():
            for key in subsection:
                default_key = default[section.name][subsection.name][key]
                if subsection[key] == default_key:
                    subsection.pop(key)
            if len(subsection) == 0: subsection.parent.pop(subsection.name)
        if len(section) == 0: section.parent.pop(section.name)
    GuiConfig.remove_comments(myconfig)
    myconfig.write()

def read_config(filename, default=None):
    '''
        Reads a configuration csv-file.
        \return structured dictionary
    '''
    if default is None:
        myconfig = configobj.ConfigObj(get_default_filename())
    else:
        myconfig = copy.deepcopy(default)
    myconfig.filename = filename
    sparse = configobj.ConfigObj(filename)
    for section in myconfig:
        if not sparse.has_key(section): continue
        for subsection in myconfig[section]:
            if not sparse[section].has_key(subsection): continue
            myconfig[section][subsection].update(sparse[section][subsection])
    myconfig.walk(convert_path, raise_errors=True, call_on_sections=False)

    return myconfig


def import_config(filename, config={}):
    """
        Imports a configuration file as flat dictionary.
        \return flat dictionary
    """
    myconfig = read_config(filename)

    # loop over all entries and flatten keys
    for section in myconfig.sections:
        for subsection in myconfig[section].sections:
            for key in myconfig[section][subsection]:
                value = myconfig[section][subsection][key]

                if key.startswith('nt') or key.startswith('posix'):
                    if os.name in key: key = key.replace(os.name + '_', '')
                    else: continue

                if value == '': value = None
                flat_key = subsection + '/' + key
                try:
                    value = eval(value)
                except:
                    pass
                config[flat_key] = value

    # turn relative to absolute paths
    config['path/root'] = get_root()
    for entry in ['run', 'data', 'r', 'viewer']:
        if not os.path.isabs(config['path/' + entry]):
            config['path/' + entry] = os.path.join(config['path/root'], os.path.normpath(config['path/' + entry]))
    if not os.path.isabs(config['data/csv_file']):
        config['data/csv_file'] = os.path.join(config['path/data'], os.path.normpath(config['data/csv_file']))

    # turn string into types
    if config['smc/binary_model'] == 'product':
        config['smc/binary_model'] = ProductBinary
    if config['smc/binary_model'] == 'logistic':
        config['smc/binary_model'] = LogisticCondBinary

    # Update configuration dictionary.
    config['run/file'] = filename
    config['run/name'] = os.path.splitext(os.path.basename(filename))[0]
    config['run/folder'] = os.path.join(config['path/run'], config['run/name'])
    config['eval/file'] = os.path.join(config['run/folder'], config['run/name'], filename)
    config['log_id'] = '0'

    return config


def import_data(config):
    """ 
        Reads a csv data file and adds the posterior distribution to the
        parameters.
        \param config configuration dictionary
    """

    # open csv file
    filename = config['data/csv_file']
    if not filename[-4:].lower() == '.csv': filename += '.csv'
    reader = csv.reader(open(filename, 'rU'), delimiter=',')
    header = reader.next()
    d = len(header)

    # read explained variable position
    if not isinstance(config['data/explained'], str): Y_pos = config['data/explained'] - 1
    else: Y_pos = header.index(config['data/explained'])

    # read predictor positions
    free_index = list()
    if not config['data/free_predictors'] is None:
        for free_range in str(config['data/free_predictors']).split('+'):
            free_range = free_range.split(':')
            if len(free_range) == 1:free_range += [free_range[0]]
            for i in xrange(2):
                if free_range[i].isdigit(): free_range[i] = int(free_range[i]) - 1
                elif free_range[i] == 'inf': free_range[i] = d - 1
                else: free_range[i] = header.index(free_range[i])
            free_index += range(free_range[0], free_range[1] + 1)

    # read principal components positions
    static_index = list()
    if not config['data/static_predictors'] is None:
        for static_range in str(config['data/static_predictors']).split('+'):
            static_range = static_range.split(':')
            if len(static_range) == 1:static_range += [static_range[0]]
            for i in xrange(2):
                if static_range[i].isdigit(): static_range[i] = int(static_range[i]) - 1
                elif static_range[i] == 'inf': static_range[i] = d - 1
                else: static_range[i] = header.index(static_range[i])
            static_index += range(static_range[0], static_range[1] + 1)
    config['data/static'] = len(static_index)

    # convert csv data to numpy.array
    sample = list()
    for row in reader:
        if len(row) > 0 and not row[Y_pos] == 'NA':
            sample += [numpy.array([
                eval(x) for x in
                    [row[Y_pos]] + # observation column
                    [row[i] for i in static_index] + # static predictor columns
                    [row[i] for i in free_index]     # free predictor columns
                ])]
    sample = numpy.array(sample)

    # cut number of observations
    config['data/max_obs'] = min(config['data/max_obs'], sample.shape[0])
    sample = sample[:config['data/max_obs'], :]

    free_header, static_header = [header[i] for i in free_index], [header[i] for i in static_index]

    # for each interaction column store the columns of the two main effects
    config['data/constraints'] = numpy.array([])
    if config['data/main_effects']:
        config['data/constraints'] = numpy.array(
                [[free_header.index(icol[0]), free_header.index(icol[1]), free_header.index(icol[2])]
                   for icol in [[col] + col.split('.x.')
                                for col in free_header if '.x.' in col] if not icol[1] == icol[2]])

    # overwrite data dependent place holders
    if config['prior/model_inclprob'] is None:
        config['prior/model_inclprob'] = 0.5
        config['prior/model_maxsize'] = config['data/max_obs']
    if config['prior/model_maxsize'] in ['n', '', None]:
        config['prior/model_maxsize'] = config['data/max_obs']
    if config['prior/var_dispersion'] in ['n', '', None]:
        config['prior/var_dispersion'] = config['data/max_obs']

    if config['prior/criterion'].lower() == 'bayes':
        Posterior = PosteriorBVS
    else:
        Posterior = PosteriorML

    config.update({'data/header' : header,
                   'data/free_header' : free_header,
                   'data/static_header' : static_header,
                   'f': Posterior(y=sample[:, 0], Z=sample[:, 1:], config=config)})
    return config

'''

def readGroups(v):
    """ 
        Reads the data file and a group file to set up a random effect model.
        \param v parameters
    """
    # open the data file to load the marker positions
    DATA_FILE = os.path.join(v['SYS_ROOT'], v['DATA_PATH'], v['DATA_DATA_FILE'])
    if not DATA_FILE[-4:].lower() == '.csv': DATA_FILE += '.csv'
    dreader = csv.reader(open(DATA_FILE, 'rU'), delimiter=',')
    DATA_HEADER = dreader.next()

    # open the group file to load all group information
    GROUP_FILE = os.path.join(v['SYS_ROOT'], v['DATA_PATH'], v['DATA_GROUP_FILE'])
    if not GROUP_FILE[-4:].lower() == '.csv': GROUP_FILE += '.csv'
    greader = csv.reader(open(GROUP_FILE, 'rU'), delimiter=',')
    greader.next()
    GROUPS_HEADER = list()
    GROUPS_ALL = dict()
    for g in greader:
        if not len(g) > 0: continue
        GROUPS_HEADER += [g[0]]
        GROUPS_ALL.update({g[0]:{'start':g[1], 'end':g[2]}})

    # pick groups from setup file
    gindex = list()
    for grange in v['DATA_GROUPS'].split('+'):
        grange = grange.split(':')
        if len(grange) == 1:grange += [grange[0]]
        for i in xrange(2):
            if grange[i].isdigit(): grange[i] = int(grange[i]) - 1
            elif grange[i] == 'inf': grange[i] = len(GROUPS_HEADER) - 1
            else: grange[i] = GROUPS_HEADER.index(grange[i])
        gindex += range(grange[0], grange[1] + 1)

    GROUPS_HEADER = [GROUPS_HEADER[i] for i in gindex]
    GROUPS = list()
    for group in GROUPS_HEADER:
        try:
            GROUPS += [{'start':DATA_HEADER.index(GROUPS_ALL[group]['start']), 'end':DATA_HEADER.index(GROUPS_ALL[group]['end'])}]
        except:
            print "Covariate %s in group %s was not found. Aborted." % (GROUPS_ALL[group]['start'], group)
            sys.exit(0)

    # pick data for the groups
    cindex = list()
    for group in GROUPS: cindex += range(group['start'], group['end'] + 1)
    if not isinstance(v['DATA_EXPLAINED'], str): Y_pos = v['DATA_EXPLAINED'] - 1
    else: Y_pos = DATA_HEADER.index(v['DATA_EXPLAINED'])

    sample = numpy.array([numpy.array([eval(x) for x in [row[Y_pos]] + [row[i] for i in cindex]])
                          for row in dreader if len(row) > 0 and not row[Y_pos] == 'NA'])

    # use just the first DATA_MAX_OBS observations
    v['DATA_MAX_OBS'] = min(v['DATA_MAX_OBS'], sample.shape[0])
    sample = sample[:v['DATA_MAX_OBS'], :]

    # initialize posterior
    v.update({'GROUPS':GROUPS, 'DATA_HEADER' : GROUPS_HEADER})
    v.update({'f': binary.posterior.Posterior(Y=sample[:, 0], X=sample[:, 1:], param=v)})
    return v

'''


def convert_path(section, key):
    '''
        Adjust the path to the operating system.
    '''
    inline = section.inline_comments[key]
    if inline is None: return
    if '$file' in inline or '$path' in inline and not section[key] == '':
        section[key] = os.path.normpath(section[key].replace('\\', '/'))

def get_default_filename(project='ibs'):
    '''
        Find absolute path of default configuration file.
    '''
    path = os.getcwd()
    while not os.path.basename(path) == 'smcdss':
        path = os.path.abspath(os.path.join(*([path] + ['..'] * 1)))
    return os.path.join(path, 'config_%s.ini' % project)

def get_root(path=os.getcwd()):
    '''
        Find root.
        \return smcdss directory.
    '''
    while not os.path.basename(path) == 'smcdss':
        path = os.path.abspath(os.path.join(*([path] + ['..'] * 1)))
    return path

def relpath(target, base=os.curdir):
    '''
        Return a relative path to the target from either the current dir or an
        optional base dir. Base can be a directory specified either as absolute
        or relative to current dir.
    '''

    if not os.path.exists(target):
        raise OSError, 'Target does not exist: ' + target

    if not os.path.isdir(base):
        raise OSError, 'Base is not a directory or does not exist: ' + base

    base_list = (os.path.abspath(base)).split(os.sep)
    target_list = (os.path.abspath(target)).split(os.sep)

    # On the windows platform the target may be on a completely different drive from the base.
    if os.name in ['nt', 'dos', 'os2'] and base_list[0] <> target_list[0]:
        raise OSError, 'Target is on a different drive to base. Target: ' + target_list[0].upper() + ', base: ' + base_list[0].upper()

    # Starting from the file path root, work out how much of the file path is
    # shared by base and target.
    for i in range(min(len(base_list), len(target_list))):
        if base_list[i] <> target_list[i]: break
    else:
        # If we broke out of the loop, i is pointing to the first differing path elements.
        # If we didn't break out of the loop, i is pointing to identical path elements.
        # Increment i so that in all cases it points to the first differing path elements.
        i += 1

    rel_list = [os.pardir] * (len(base_list) - i) + target_list[i:]
    if len(rel_list) == 0:return ''
    return os.path.join(*rel_list)
