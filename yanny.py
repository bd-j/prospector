"""
Module:
    yanny

Purpose:

    tools for working with Yanny parameter files, aka ftcl parameter files.
    The file format is described here

        http://www.sdss.org/dr7/dm/flatFiles/yanny.html

    Limitations:
        1) Currently only supports reading.

        2) Because numpy arrays are fixed length, all char fields must be
        declared with an explicit length.

        3) enums are not yet supported.

Functions:
    read:  
        A convenience function to read from a yanny parameter file. See docs
        for this function for full info.
    readone:
        Same as read() but with the keyword one=True, returning the first
        structure rather than a dictionary.

Classes:
    Class Name:
        Yanny
    Purpose:
        Read and write yanny parameter files.
    Useful Methods:
        read():  Read data from the file.  Subsets of the data can be returned. See
            docs for the read() method for more info.
        open(): Open new file.  This is also called on construction.
        seek(pos): Seek to the position in the file.

Modification History:
    Created: 2010-04-07, Erin Sheldon, BNL
"""
import os
import sys
from sys import stdout
import re

import numpy


# keys are always lower case
_yanny2numpy  = \
        {'char': 'S',
         'short': 'i2',
         'int': 'i4',
         'long': 'i8',
         'float': 'f4',
         'double': 'f8'}
_numpy2yanny=\
         {'s':  'char',
         'i2': 'short',
         'i4': 'int',
         'i8': 'long',
         'f4': 'float',
         'f8': 'double'}


badbracket_reg=re.compile('\{ .*\{ .*\} .*\}')


def readone(fname, names=None, indices=None, getpars=False, defchar=255, 
            verbose=False):
    """
    Module:
        yanny
    Name:
        readone
    Purpose:

        Simple wrapper for yanny.read with the keyword one=True.  The return
        value for one=True is the first structure instead of a dictionary.  If
        getpars=True it is (firststruct,pars)

        See docs for yanny.read for more info.
 
    """

    y = Yanny(fname, defchar=defchar, verbose=verbose)
    data = y.read(one=True, names=names, indices=indices, getpars=getpars)
    del y
    return data

def read(fname, one=False, names=None, indices=None, getpars=False, defchar=255,
         verbose=False):
    """
    Module:
        yanny
    Name:
        read
    Purpose:

        Read data from a yanny parameter file, as described here
        
            http://www.sdss.org/dr7/dm/flatFiles/yanny.html

        This is a convenience function; under the hood, a Yanny object is
        instantiated and used for reading.  The result is a dictionary keyed by
        the structure names, if any structures are found.   If names or indices
        are sent, subsets are returned.  If the one=True keyword is sent, the
        first structure is returned by itself, not wrapped in a dictionary.  If
        getpars=True, a tuple of (structs, pars) are returned.

        Limitations:
            1) Currently only supports reading.

            2) Because numpy arrays are fixed length, all char fields must be
            declared with an explicit length.

            3) enums are not yet supported.

    Calling Sequence:
        from sdsspy import yanny
        result = yanny.read(fname, one=False, names=None, indices=None, getpars=False)

    Inputs:
        fname: The file name.
    
    Optional Inputs:
        one: 
            Return the first struct.  Normally the result is a dictionary keyed
            by structure name, in this case just the structure is returned.
            Can be used in conjunction with names or indices
        names: 
            A name or sequence of names, representing a subset of structures to
            return.
        indices: 
            An indices or sequence if indices representing a subset of the
            structures to return.
        getpars:  If True, return a tuple (structs, pars).

    Outputs:
        Normal output is a dictionary keyed by structure name.  Each value is
        then a numerical python array with fields (aka recarray, structure)
        representing eacah.

        If one=True, the first struct is returned, not wrapped by a dictionary. 

        If getpars=True, a tuple (structs, pars) is returned.

    Example:
        # an example .par file
        name John Smith
        age 35

        typedef struct {
          char flag[20];  # Flag name
          short bit;      # Bit number, 0-indexed
          char label[20]; # Bit label
        } maskbits;

        typedef struct {
          char flag[20];  # Flag name
          short datatype; # Data type {8, 16, 32, 64}
        } masktype;

        masktype SPPIXMASK 32
        # The following mask bits are for the fiber, set in FIBERMASK_BITS()
        maskbits SPPIXMASK  0 NOPLUG          # Fiber not listed in plugmap file
        maskbits SPPIXMASK  1 BADTRACE        # Bad trace from routine TRACE320CRUDE
        maskbits SPPIXMASK  \\
                2 BADFLAT         # Low counts in fiberflat

        masktype TARGET 32
        maskbits TARGET  0 QSO_HIZ
        maskbits TARGET  1 QSO_CAP
        maskbits TARGET  2 QSO_SKIRT


        # get all structs in a dict
        >>> import sdsspy
        >>> structs = sdsspy.yanny.read('example.par')
        >>> structs.keys()
        ['maskbits','masktype']
        >>> structs['maskbits'].dtype.descr
        [('flag', '|S20'), ('bit', '<i2'), ('label', '|S20')]
        >>> structs['maskbits']['flag']
        array(['SPPIXMASK', 'SPPIXMASK', 'SPPIXMASK', 'TARGET', 'TARGET', 'TARGET'],
              dtype='|S20')

        # Also get pars
        >>> structs, pars = sdsspy.yanny.read('example.par',getpars=True)
        >>> pars
        {'age': 35, 'name': 'John Smith'}

        # Get the first struct
        >>> maskbits = sdss.yanny.read('example.par',one=True)
        >>> maskbits['bit']
        >>> array([0, 1, 2, 0, 1, 2], dtype=int16)

    Modification History:
        Created: 2010-04-07, Erin Sheldon, BNL

    """

    y = Yanny(fname, defchar=defchar, verbose=verbose)
    data = y.read(one=one, names=names, indices=indices, getpars=getpars)
    del y
    return data

class Yanny():
    """
    Class:
        Yanny
    Purpose:
        A class for working with Yanny parameter files, aka ftcl parameter
        files.

            http://www.sdss.org/dr7/dm/flatFiles/yanny.html

        Limitations:
            1) Currently only supports reading.

            2) Because numpy arrays are fixed length, all char fields must be
            declared with an explicit length.

            3) enums are not yet supported.

    Construction:
        import sdsspy.yanny
        y = sdsspy.yanny.Yanny(fname=None, mode='r', verbose=False)

        or

        y=sdsspy.yanny.Yanny()
        y.open(fname=None,mode='r')

    Reading a file:
        result = y.read(one=False, names=None, indices=None, getpars=False)
        
        See docs for the read() method for more details.

    Useful Methods: (see individual methods for more details)
        read(): Read the data.
        open(): Open a file.  Also called on construction.
        seek(pos): Seek to the position in the file.


    Modification History:
        Created: 2010-04-07, Erin Sheldon, BNL
    """
    def __init__(self, fname=None, mode='r', defchar=255, verbose=False):
        self._fobj=None

        self._fname = fname
        self._mode = mode

        self.defchar = defchar

        self.verbose=verbose

        self.open(self._fname, self._mode)

    def open(self, fname=None, mode='r'):
        if isinstance(self._fobj, file):
            self._fobj.close()

        self._fname = fname
        self._fobj=None
        if self._fname is not None:
            self._fname = os.path.expanduser(self._fname)
            self._fname = os.path.expandvars(self._fname)
            self._fobj = open(self._fname, mode)

    def seek(self, pos):
        """
        seek(pos): Seek to the indicated postion in the open file.
        """
        if not isinstance( self._fobj, file):
            raise ValueError("No file is open")
        if self._fobj.closed:
            raise ValueError("No file is open")
        self._fobj.seek(pos)


    def read(self, one=False, names=None, indices=None, getpars=False):
        """
        Class:
            Yanny
        Method Name:
            read
        Purpose:

            Read data from a yanny parameter file, as described here
            
                http://www.sdss.org/dr7/dm/flatFiles/yanny.html

            The result is a dictionary keyed by the structure names, if any
            structures are found.   If names or indices are sent, subsets are
            returned.  If the one=True keyword is sent, the first structure is
            returned by itself, not wrapped in a dictionary.  If getpars=True,
            a tuple of (structs, pars) are returned.

            Limitations:
                1) Currently only supports reading.

                2) Because numpy arrays are fixed length, all char fields must be
                declared with an explicit length.

                3) enums are not yet supported.

        Calling Sequence:
            from sdsspy import yanny
            y=yanny.Yanny(fname, mode='r', verbose=False)
            result = y.read(one=False, names=None, indices=None, getpars=False)
            
            # by default all data are read

        Optional Inputs:
            one: 
                Return the first struct.  Normally the result is a dictionary keyed
                by structure name, in this case just the structure is returned.
                Can be used in conjunction with names or indices
            names: 
                A name or sequence of names, representing a subset of structures to
                return.
            indices: 
                An indices or sequence if indices representing a subset of the
                structures to return.
            getpars:  If True, return a tuple (structs, pars).

        Outputs:
            Normal output is a dictionary keyed by structure name.  Each value is
            then a numerical python array with fields (aka recarray, structure)
            representing eacah.

            If one=True, the first struct is returned, not wrapped by a dictionary. 

            If getpars=True, a tuple (structs, pars) is returned.

        Example:
            # an example .par file
            name John Smith
            age 35

            typedef struct {
              char flag[20];  # Flag name
              short bit;      # Bit number, 0-indexed
              char label[20]; # Bit label
            } maskbits;

            typedef struct {
              char flag[20];  # Flag name
              short datatype; # Data type {8, 16, 32, 64}
            } masktype;

            masktype SPPIXMASK 32
            # The following mask bits are for the fiber, set in FIBERMASK_BITS()
            maskbits SPPIXMASK  0 NOPLUG          # Fiber not listed in plugmap file
            maskbits SPPIXMASK  1 BADTRACE        # Bad trace from routine TRACE320CRUDE
            maskbits SPPIXMASK  \\
                    2 BADFLAT         # Low counts in fiberflat

            masktype TARGET 32
            maskbits TARGET  0 QSO_HIZ
            maskbits TARGET  1 QSO_CAP
            maskbits TARGET  2 QSO_SKIRT


            # get all structs in a dict
            >>> from sdsspy import yanny
            >>> y=yanny.Yanny('example.par')
            >>> structs = y.read()
            >>> structs.keys()
            ['maskbits','masktype']
            >>> structs['maskbits'].dtype.descr
            [('flag', '|S20'), ('bit', '<i2'), ('label', '|S20')]
            >>> structs['maskbits']['flag']
            array(['SPPIXMASK', 'SPPIXMASK', 'SPPIXMASK', 'TARGET', 'TARGET', 'TARGET'],
                  dtype='|S20')

            # Also get pars
            >>> y.seek(0)
            >>> structs, pars = y.read(getpars=True)
            >>> pars
            {'age': 35, 'name': 'John Smith'}

            # Get the first struct
            >>> y.seek(0)
            >>> maskbits = y.read(one=True)
            >>> maskbits['bit']
            array([0, 1, 2, 0, 1, 2], dtype=int16)

        Modification History:
            Created: 2010-04-07, Erin Sheldon, BNL
        """

        # get all the info we can.  nameordered has the names ordered as
        # the structdefs were found in the file
        structs, pars, nameordered = self.readall(nameordered=True)


        if indices is not None:
            
            indices=numpy.array(indices, ndmin=1, copy=False)

            if indices.max() > (len(nameordered)-1):
                raise ValueError("Requested indices too large: %s" % indices.max())

            # add unique check
            outstructs = {}
            for i in indices:
                name = nameordered[i]
                outstructs[name] = structs[name]

                if one:
                    outstructs = outstructs[name]
                    break

        elif names is not None:
            if not isinstance(names, (tuple,list,numpy.ndarray)):
                names=[names]

            outstructs = {}
            for name in names:
                if name not in structs:
                    raise ValueError("Bad struct name request '%s'" % name)
                outstructs[name] = structs[name]


                # user requested just the structure of first one matching.
                if one:
                    outstructs = outstructs[name]
                    break

        elif one:
            # user requested just the structure
            outstructs = structs[nameordered[0]]
        else:
            outstructs = structs


        if getpars:
            return outstructs, pars
        else:
            return outstructs


    def readall(self, nameordered=False):
        """
        Class:
            Yanny
        Method Name:
            readall(nameordered=False)
        Purpose:
            Equivalent to read(pars=True), returning structs,pars

            If nameordered=True, then the full tuple returned is
                (structs, pars, nameordered)

            Where namerodered is the list of struct names in the order they
            were found.

        """


        if self._fobj is None:
            raise RuntimeError("No file is open")
        # read all data into a stringio
        # provides line-by-line processing but faster if we decide to
        # run back through
        # decided to just keep the lines in memory
        #self.read_to_stringio()

        # a first pass through to get the typedefs and count the number of instances
        # of each struct
        allnames = []
        structs = {}
        pars = {}
        while 1:
            
            line = self.get_line()
            if line is None:
                break


            if len(line) > 0:
                # there were non-commented characters

                # this also does a strip() on all elements
                ls = line.split()

                if ls[0].lower() == 'typedef':
                    # we found a typedef statement

                    # find opening and closing { }
                    typedef = self.get_full_typedef(line)
                    if self.verbose:
                        print 'typedef is:\n',typedef
                    name, descr = self.typedef2dtype(typedef)
                    allnames.append(name)
                    structs[name] = {}
                    structs[name]['descr'] = descr
                    structs[name]['lines'] = []
                else:

                    # see if this is a known struct
                    t = ls[0].lower()
                    if t in structs:
                        # this line is part of a struct array
                        structs[t]['lines'].append(line)

                    else:
                        # this is probably a key-value pair. don't support other
                        # types of stuff yet
                        parname = ls[0]

                        # stuff the rest into a single string
                        loc = line.find(' ')
                        if loc == -1:
                            valstring=''
                        else:
                            valstring = line[loc+1:].strip()

                        try:
                            val = eval(valstring)
                        except:
                            # failure:
                            # just glob anything after the first white space
                            # into a string
                            val = valstring
                        
                        pars[parname] = val

        if len(structs) > 0:
            # process the structs
            outstructs = self.process_structs(structs)
        else:
            outstructs = {}

        if nameordered:
            return outstructs, pars, allnames
        else:
            return outstructs, pars



    def process_structs(self, structs):
        outstructs = {}
        for name in structs:
            outstructs[name] = self.structdict2array(structs[name])
        return outstructs

    def structdict2array(self, struct):
        """
        This takes the dict with ['descr'] and ['lines'] and creates the
        numpy array with fields.
        """
        n = len( struct['lines'] )
        descr = struct['descr']
        if self.verbose:
            print 'descr:',descr
        data = numpy.zeros(n, dtype=descr)
        # now run through the lines and interpret in terms of the given data type

        i=0
        for line in struct['lines']:

            # this descr should not have big/little endian in
            # the type
            data[i] = self.structline2array(descr, line)

            i += 1

        return data

    def structline2array(self, descr, line):
        """
        Convert a single struct line into a single numpy rec row
        """

        # get words, respecting things in " as single word
        words = self.structline2words(line)

        #print 'data.dtype.descr:',descr
        data = numpy.zeros(1, dtype=descr)

        #print 'data:',data
        # loop over and copy in words

        # this is where we are in the words
        # we skip the first word, which is the struct name
        word_index=1

        for d in descr:
            name = d[0]
            thisdata = data[name][0]

            #print 'thisdata.dtype.descr:',thisdata.dtype.descr
            #print 'thisdata:',thisdata

            if numpy.isscalar(thisdata):
                data[name] = words[word_index]
                word_index += 1
            else:
                if len(thisdata.shape) > 0:
                    # we want to reshape to a flat array to make this easier
                    thisdata = thisdata.reshape(thisdata.size)
                
                for j in range(thisdata.size):
                    # string to number conversion must happen explicitly
                    thisdata[j] = words[word_index]
                    word_index += 1

        return data

    def get_struct_value(descr1, line):
        """
        Extract a value and return it along with the rest of
        the line
        """

        nd = len(descr1)
        if nd == 2:
            type = descr1[1]
            shape=None
        elif nd == 3:
            type,shape = descr1[1],descr1[2]
        else:
            raise ValueError("Expected descr len 2 or 3, got %s" % len(descr1))


    def structline2words(self, line):
        """
        The trick here is to preserve anything inside quotes.

        For the characters not in quotes, remove the redundant { }
        """

        # first split by '"' character
        # then in every other section should be the non-quoted stuff, 
        # e.g. 0,2,4,...

        # first replace { { } } with ""

        line = badbracket_reg.sub('""', line)

        # now remove the { } from everywhere except inside quotes
        ls = line.split('"')
        lslen = len(ls)

        if lslen % 2 == 0:
            raise ValueError("unmatched quote in line: %s" % line)

        # only every other element needs to be split by white space
        # and have brackets removed
        words = []
        i=0
        for i in range(0,lslen):
            if i % 2 == 0:
                # this section was not in quotes.  Replace brackets
                # with spaces
                ls[i] = ls[i].replace('{',' ').replace('}',' ')
                words += ls[i].split()
            else:
                # this section was in quotes, just add it to words
                # without any further splitting
                words.append(ls[i])

        return words



    def read_to_stringio(self):
        import cStringIO
        data = self._fobj.read()
        self._data = cStringIO.StringIO(data)

    def get_line(self, no_end=False, verbose=False):
        """
        Get the next line, remove comments
        """
        line = self._fobj.readline()
        #line = self._data.readline()

        if len(line) == 0:
            if no_end:
                raise ValueError("Unexpected end of line")
            return None

        icomment = line.find('#')
        if icomment != -1:
            line = line[0:icomment]

        line = line.strip()

        if len(line) > 0:
            if line[-1] == '\\':
                # remove last character
                line = line[0:-1]
                # add next line
                line += self.get_line(no_end=True, verbose=False)
                line = line.strip()

        if self.verbose and verbose:
            print "line: '%s'" % line
        return line



    def yanny_type_to_numpy_type(self, yanny_type):
        yanny_type = yanny_type.lower()
        if yanny_type not in _yanny2numpy:
            raise ValueError("invalid yanny type: '%s'" % yanny_type)

        return _yanny2numpy[yanny_type]

    def numpy_type_to_yanny_type(self, numpy_type):
        numpy_type = numpy_type.lower()
        if numpy_type not in _numpy2yanny:
            raise ValueError("invalid numpy type: '%s'" % numpy_type)

        return _numpy2yanny[numpy_type]



    def get_yanny_string_len_dims(self, chardims):
        """
        char always has lenght [num] but can also have dims beyond
        that.  E.g. a 2x3 array of length 20 strings would be declared
        as 
            char name[20][2][3]

        """

        cs = chardims.split('][')
        if len(cs) == 1:
            slen = chardims.replace(']','').replace('[','')
            dims=None
        else:
            # last one is the string length
            slen = cs[-1].replace(']','')

            dims = ']['.join(cs[0:-1])+']'

        return slen, dims



    def get_yanny_def_as_descr(self, yanny_def):

        ys = yanny_def.split()

        yanny_type = ys[0]
        numpy_type = self.yanny_type_to_numpy_type(yanny_type)
        fieldname = ys[1]
        

        left_loc = fieldname.find('[')
        right_loc = fieldname.rfind(']')

        if left_loc == -1:
            if numpy_type == 'S':
                raise ValueError("No length given for char field: '%s'" % yanny_def)
            name = fieldname
            shape = None
        else:
            # make sure we have a match
            if right_loc == -1:
                raise ValueError("no matching ] for field def: '%s'" % yanny_name)

            name = fieldname[0:left_loc]
            dims = fieldname[left_loc:right_loc+1]

            if numpy_type == 'S':
                slen, dims = self.get_yanny_string_len_dims(dims)

                if slen == '':
                    #raise ValueError("char field declared with no size: '%s'" % fieldname)
                    slen = str(self.defchar)
                numpy_type += slen

            # if 'S' we may still end up with no dims
            if dims is None:
                shape = None
            else:
                shape = dims.replace('][',',').replace('[','(').replace(']',')')

                if shape.find(',') == -1:
                    # not multi-dimensional
                    shape = shape.replace('(','').replace(')','')
                # convert either to a number or tuple thereof
                shape = eval(shape)

        if shape is not None:
            descr = (name, numpy_type, shape)
        else:
            descr = (name, numpy_type)
        return descr

    def typedef2dtype(self, typedef):
        """
        Convert the typedef into a numpy type descriptor
        """

        ts = typedef.split()
        if ts[0].lower() != 'typedef':
            raise ValueError("Bad typedef: '%s'" % typedef)
        if ts[1].lower() != 'struct':
            raise ValueError("Bad typedef: '%s'" % typedef)

        # get the part between the { }
        left_loc = typedef.find('{')
        right_loc = typedef.find('}')

        if left_loc == -1 or right_loc == -1:
            raise ValueError("Bad typedef: '%s'" % typedef)

        body = typedef[left_loc+1:right_loc]

        # split on ; to get the field defs
        field_defs = [f for f in body.split(';') if f != '']

        # now look at the rest to get the name
        name = typedef[right_loc+1:].lower().strip()
        if name[-1] == ';':
            name = name[0:-1]

         
        if self.verbose:
            print "struct name: '%s'" % name
            print "field defs:"
        descr = []
        for f in field_defs:
            d = self.get_yanny_def_as_descr(f)
            if self.verbose:
                print '  ',f,'descr:',d

            descr.append(d)

        return name, descr


    def get_full_typedef(self, current_line):
        """
        return the typedef name and a dictionary keyed off the typedef name 
            1) the name of the typedef
            2)
        """

        tdef = ''
        while 1:

            loc = current_line.find('}')
            if loc == -1:
                # end } of typedef not on this line, concatenate to tdef 
                tdef += current_line
                current_line = self.get_line(no_end=True)
            else:
                # we found the ending }, now look for ending "name;"
                tdef += current_line[0:loc]
                rest = current_line[loc:]

                # now keep going until we get to the semicolon
                sloc = rest.find(';')
                while sloc == -1:
                    tdef += rest

                    rest = self.get_line(no_end=True)
                    sloc = rest.find(';')

                # we will throw away the rest of this line
                tdef += rest[0:sloc]
                break

            #print 'tdef: ',tdef

        return tdef


                



