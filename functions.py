from itertools import count
from pydub import AudioSegment
import os
import array
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# TODO:  
# Grab frequency generator function to determine note offset from theory
# look how changes affect the overtone series
# Compare with sound from good violins
# Record playing for a few hours to check when I fuck up what happens to the spectrum
# Check how differing playing postions change the overtone series
# I assume more clearly delimited overtones sound better? Honestly dont know
# May just be the way I play
class Note:
    """
    Note as recorded from the violin
    """
    def __init__(self, name, path) -> None:
        """
        name --> Name of the note
        path --> Path to audiofile
        """
        self.name = name
        self.path = path
        self._get_fft()

    def _get_fft():
        """
        Get file from self.path get data 
        """
        A4 = AudioSegment.from_wav(self.path)
        self.A4 = A4[0.1*len(A4):0.9*len(A4):]
        
        # Mutlitple channels first we split
        channels = self.A4.split_to_mono()
        for i in channels:
            array_data = np.array(i.get_array_of_samples()).T.astype(np.float32)

            fft_data = np.fft.fft(array_data,norm=None)
            if not hasattr(self, 'fit_data'):
                self.fft_data = fft_data
            else:
                self.fft_data += fft_data

        return None


    def mod_spectrum(freq1, freq2, factor):
        '''
        Modify frequency range by factor
        '''
        bound_0 = freq1*self.A4.frame_rate
        bound_1 = freq2*self.A4.frame_rate
        self.fft_data[bound_0:bound_1:] = self.fft_data[bound_0:bound_1:]*factor

        return None

    def make_plot_fft():
        fig = plt.figure(figsize=(16,16))
        # FFT is currnetly in units of array items 1 array item corresponds to a timestep of 1/sample_rate so used numpy fftfreq to get unitless timesteps
        plt.plot(np.fft.fftfreq(len(self.fft_data))*sample_rate,fft_data,label='Data')
        #plt.vlines(270,-1,10000, color='black',label="Helmholtz")
        #plt.vlines(450,-1,10000, color='black',label="Corpus")
        #plt.vlines(550,-1,10000, color='black',label="Corpus")
        #plt.vlines(700,-1,10000, color='black',label="nasal")
        #plt.vlines(1000,-1,10000,color='black', label="nasal")
        #plt.vlines(2000,-1,10000,color='black', label="higher")
        #plt.vlines(3500,-1,10000,color='black', label="higher")
        plt.xscale('log')
        plt.legend()
        plt.show()

        return None

    def make_audio(name, path=None):
        """
        Makes wav file from fft
        """
        array_data_mod = array.array(self.A4.array_type, np.abs(np.fft.ifft(self.fft_data)).astype(np.int))
        
        A4_mod = self.A4._spawn(array_data_mod)
        if path is not None:
            filepath = os.path.join(path,name)
        else:
            filepath = os.path.join(os.path.dirname(self.path),name)
        A4_mod.export(filepath, format="wav")

        return None



class Violin:
    """
    Class to store all notes of a Violin
    Can make modifications to all notes
    """
    def __init__(note_dict, name=None):
        '''
        note_dict -- Dictionary of Notes class notes
                    - Will be used to populate all notes through interpolation
        '''
        self.notes = OrderedDict()
        #Iterate all notes
        notes = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#']
        count = 0
        octave = 3
        while name != 'E7':
            if notes[count%12]=='C': octave += 1
            name = notes[count%12]+str(octave)
            if hasattr(notes_dict, name):
                self.notes[name] = note_dict[name]
            else:
                self.notes[name] = None 
                # TODO: Write some interpolation routine for this
        if name is not None: self.name = name
        else: self.name = None

    def mod_spectrum(freq1, freq2, factor, new_name= None):
        '''
        Modify frequency range by factor
        '''
        for key in self.notes:
            self.notes[key].mod_spectrum(freq1, freq2, factor)
        if new_name is not None: self.name = new_name
        
        return None

    def make_interpolation():
        raise Exception('To be implemented')

    def make_scales(path=None, name=None):
        """
        Makes C and F# chromatic scale progressions
        """
        # Make all the wav files
        if path is None:
            path = os.getcwd()
        if name is None:
            name = 'Violin'

        C_scale = AudioSegment.empty()
        for key in self.notes:
            if any([i in key for i in ['C','D','E','F','G','A','B']]):
                mid_length = len(self.notes[key])//2
                C_scale += self.notes[key][mid_length-500:mid_length+500:]
            
        Fs_scale = AudioSegment.empty()
        for key in self.notes:
            if any([i in key for i in ['C#','D#','F','F#','G#','A#','C']]):
                mid_length = len(self.notes[key])//2
                Fs_scale += self.notes[key][mid_length-500:mid_length+500:]

        C_scale.make_audio(name=name+'_C_scale.wav' , path=path)
        Fs_scale.make_audio(name=name+'_F#_scale.wav' , path=path)

        return None

    def scale_to_sound(dict_to_fit):
        """
        Creates scaling function based on individual notes fit
        
        TODO: Make plotting of this with the addition of general changes that can be induced by changing playing behavior and build
        """ 
        scalings = {}
        for key in dict_to_fit:
            scalings[key]=dict_to_fit[key].fft_data/self.notes[key].fft_data

        return scalings





class Song:
    """
    Song instance
    to make music sound like my instrument
    """
    
    measures = []
    def __init__(self,base_note=4,bpm=60, bpb=4,key_signature = 'C') -> None:
        self.base_note = base_note # base note 4 == a fourth
        self.bpm = bpm # beats per minute
        self.bpb = bpb # beats per bar
        self.key_signature = key_signature
        self.length = 0
        pass

    def add_measure(measure):
        self.length += measure.length
        self.measures += measure

    def change_key(key=None):
        if key is not None:
            self.key_signature = key

        for measure in self.measures:
            measure.change_key(key=self.key_signature)

    def play_with_instrument(instrument,path,name):
        """
        instrument class -- violin
        makes file and saves the audio
        """
        song = AudioSegment.empty()
        for measure in self.measures:
            for note in measure.notes:
                mid_length = len(instrument.notes[note.name])//2
                length = note.length*self.bpm/self.bpb /2
                song += instrument.notes[note.name][mid_length-length:mid_length+length:]

        song.make_audio(name=name , path=path)


class Measure:
    def __init__(self,bpb) -> None:
        self.bpb = bpb
        self.notes = []
        self.length= 0
        pass

    def add_note(note):
        """
        note class containing info about the note / pause
        """
        if type(note) != list:
            note = list(note)
        for i in note:
            # Check there is enough time in the bar
            if self.length+i.length <= self.bpb:
                self.is.append(i)
                self.length += i.length # TODO: Check that note duration is consistent 

            else:
                raise Exception('Notes too long for measure!')
            return self

    def change_key(key):
        """
        key signature as returned by KeySignature
        """
        for i in key.to_change:
            if i in self.notes:
                indeces = self.notes.index(i)
                for j in indeces:
                    self.notes[j] = key.change_to[key.to_change.index(i)]
        return None 


class KeySignature:
    """Class to keep track of what each signature requires to reduce the
    amount of computation"""
    def __init__(self,key) -> None:
        # circle of fifths going clockwise and anticlockwise respectively
        cofs = ['G','D','A','E','B#','G#','C#']
        # Associated sharps
        fss = ['F','C','G','D','A','E','B']
        fst = ['F#','C#','G#','D#','A#','F','C']
        coff = ['F','A#',"D#",'G#']
        # Associated flats
        fff = ['B', 'E', 'A','D']
        fft = ['A#', 'D#', 'G#','C#']
        if key in cofs:
            index = cofs.index(key)
            self.to_change = fss[:index:]
            self.change_to = fst[:index:]

        elif key in coff:
            index = coff.index(key)
            self.to_change = fff[:index:]
            self.change_to = fft[:index:]




class Note:
    """
    Notes to be used to construct songs, 
    all songs should be written in th key of C and then later automatically
    transposed by the Song class
    """
    def __init__(self,length, name) -> None:
        self.length = length
        self.name = name
        
        pass




def OdetoJoy():
    """Returns song instance"""
    key = KeySignature(key='G')
    bpm = 140
    bpb = 4
    base_note = 1/4
    
    ode_to_joy = Song(base_note=base_note,bpm=bpm, bpb=bpb,key_signature = key)
    B4 = Note(length=1/4, name = "B4")
    C5 = Note(length=1/4, name = "C5")
    D5 = Note(length=1/4, name = "D5")
    A4 = Note(length=1/4, name = "A4")
    G4 = Note(length=1/4, name = "G4")
    B4_l = Note(length=3/8, name = "B4")
    A4_l = Note(length=3/8, name = "A4")
    A4_8th = Note(length=1/8, name = "A4")
    A4_2 = Note(length=1/2, name = "A4")
    G4_8th = Note(length=1/8, name = "G4")
    G4_2 = Note(length=1/2, name = "G4")
    G4_full = Note(length=1, name = "G4")
    B4_8 = Note(length=1/8, name = "B4")
    C5_8 = Note(length=1/8, name = "C5")
    D4_2 = Note(length=1/2, name = "D4")


    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([B4,B4,C5,D5])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([D5,C5,B4,A4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([G4,G4,A4,B4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([B4_l,A4_8th,A4_2])
    )


    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([B4,B4,C5,D5])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([D5,C5,B4,A4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([G4,G4,A4,B4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([A4_l,G4_8th,G4_2])
    )


    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([A4,A4,B5,G5])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([A4,B4_8,C5_8,B4,G4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([A4,B4_8,C5_8,B4,A4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([G4,A4,D4_2])
    )


    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([B4,B4,C5,D5])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([D5,C5,B4,A4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([G4,G4,A4,B4])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([A4_l,G4_8th,G4_2])
    )
    ode_to_joy.add_measure(
        Measure(bpb=4).add_note([G4_full])
    )

    ode_to_joy.change_key()

    return ode_to_joy