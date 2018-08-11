import pandas as pd
import numpy as np
import librosa
import pickle
import collections
import copy
import matplotlib.pyplot as plt
import matplotlib.style as ms
import seaborn as sns
ms.use('seaborn-muted')
from sklearn.preprocessing import StandardScaler




# TODO: I'm going to write better comments and documentation.

#### DEFN and UTILS ####

# For tones > 12, drop their octave so that tone becomes same note in range [0-12]
def reduce_oct(tone_or_tones):
    if type(tone_or_tones) == tuple:
        return tuple([note%12 if note%12 else 12 for note in tone_or_tones])
    elif type(tone_or_tones) == list:
        return [note%12 if note%12 else 12 for note in tone_or_tones]
    elif type(tone_or_tones) == int:
        tone_or_tones = tone_or_tones%12 if tone_or_tones%12 else 12
        if tone_or_tones > 0:
            return tone_or_tones
        else:
            return 13+tone_or_tones
    else:
        return None

def transpose_tone(tone, by):
    return reduce_oct(tone + by)

def transpose_tones(tones, by):
    return tuple([transpose_tone(tone, by) for tone in tones])

TONES_TO_STR={1:'C',
              2:'C#',
              3:'D',
              4:'D#',
              5:'E',
              6:'F',
              7:'F#',
              8:'G',
              9:'G#',
              10:'A',
              11:'A#',
              12:'B'}

def def_STRS_TO_TONE():
    #{'C': 1,... 'B':12}
    STRS_TO_TONE = {}
    for tone in TONES_TO_STR.keys():
        note = TONES_TO_STR[tone]
        STRS_TO_TONE[note] = tone
    return STRS_TO_TONE
STRS_TO_TONE = def_STRS_TO_TONE()

# Canonical chord tones
CHORD_TYPES = {'maj'      :(1,5,8),    # Major
               'min'      :(1,4,8)     # Minor
               #'dim'      :(1,4,7)    # Diminished
               #'dim7':(1,4,7,11)
               #'maj7'     :(1,5,8,11)
}

def _defineCHORDS_abs():
# Define absolute chords by each root tone's root position chords
# i.e. {'C':(1,5,8), 'c':(1,4,8)...}
    CHORDS_abs = {}
    TONES_TO_CHORDS_abs ={}
    canonical_maj = CHORD_TYPES['maj']
    canonical_min = CHORD_TYPES['min']
    for root_tone in range(1,13):
        root_note = TONES_TO_STR[root_tone]
        maj_tones = transpose_tones(canonical_maj, root_tone - 1)  # Tone's major chord
        min_tones = transpose_tones(canonical_min, root_tone - 1)  # Tone's minor chord

        CHORDS_abs[root_note.upper()+'M'] = maj_tones
        CHORDS_abs[root_note.lower()+'m'] = min_tones
        TONES_TO_CHORDS_abs[maj_tones] = root_note.upper()+'M'
        TONES_TO_CHORDS_abs[min_tones] = root_note.lower()+'m'
    return CHORDS_abs, TONES_TO_CHORDS_abs
CHORDS_abs, TONES_TO_CHORDS_abs = _defineCHORDS_abs()

# Chords whose tones are sorted on a C-scale
# i.e {'C':(1,5,8)...'g':(3,7,12)}
# i.e. {(1,5,8):'C'...(3,7,12):'g'}
def _defineCANONICAL_CHORD_TONES():
    CANONICAL_CHORD_TONES = {}
    CANONICAL_TONES_TO_CHORD = {}
    for chord, tones in CHORDS_abs.items():
        canonical_tones = tuple(sorted(tones))
        CANONICAL_CHORD_TONES[chord] = canonical_tones
        CANONICAL_TONES_TO_CHORD[canonical_tones] = chord
    return CANONICAL_CHORD_TONES, CANONICAL_TONES_TO_CHORD
CANONICAL_CHORD_TONES, CANONICAL_TONES_TO_CHORD = _defineCANONICAL_CHORD_TONES()

# Canonical circle-of-fifths of distances from C.
# Derivation:
    #posdist = [(C + trans, trans/7) for trans in range(0,7*7,7)]
    #negdist = [(C - trans, trans/7) for trans in range(0,7*7,7)]
    #distances = sorted(posdist+ negdist)
DIST_OF_TONE = {1:0,
                2:5,
                3:2,
                4:3,
                5:4,
                6:1,
                7:6,
                8:1,
                9:4,
                10:3,
                11:2,
                12:5}

MAJ_SCALE = (1,3,5,6,8,10,12)
MIN_SCALE = (1,3,4,6,8,9,11,12)

HARMONY_DICT = {
    'Tones':       {'I':(1,5,8),'II':(3,7,10),'III':(5,9,12),'IV':(1,6,10),'V':(3,8,12),'VI':(2,5,10),'VII':(4,7,12),
                    'i':(1,4,8),'ii':(3,6,10),'iii':(5,8,12),'iv':(1,6,9),'v':(3,8,11),'vi':(1,5,10),'vii':(3,7,12),
                    'IIb':(2,6,9),'iib':(2,5,9),'IIIb':(4,8,11),'iiib':(4,7,11),'TT':(2,7,11),'tt':(2,7,10),'VIb':(1,4,9),'vib':(4,9,12),'VIIb':(3,6,11),'viib':(2,6,11)},

    'Distance':    {'I':(0,2),'II':(5,5),'III':(5,3),'IV':(2,5),'V':(1,1),'VI':(5,3),'VII':(5,5),
                    'i':(4,0),'ii':(3,3),'iii':(4,6),'iv':(5,2),'v':(5,2),'vi':(3,5),'vii':(3,6),
                    'IIb':(6,3),'iib':(7,7),'IIIb':(6,6),'iiib':(6,6),'TT':(10,10),'tt':(10,10),'VIb':(4,4),'vib':(5,6),'VIIb':(6,3),'viib':(3,3)}
                }
HARMONY = pd.DataFrame(HARMONY_DICT)
HARMONY['Chord'] = HARMONY['Tones'].apply(lambda x: CANONICAL_TONES_TO_CHORD[x])
HARMONY.reset_index(inplace=True)
HARMONY.rename({'index':'Harmony'},axis=1,inplace=True)

def lookup_tonality(tonality, by, at, to_find):
    tonality = Chord(tonality)
    at = str(at)
    raw_row = HARMONY[HARMONY[by]==at]
    raw_chord = raw_row['Chord'].tolist()[0]

    if by == 'Chord':
        transpose_for = ['Harmony', 'Distance']
        transposition = Chord(tonality).transposition_to(Chord(at))
        if Chord(at).is_major():    # Looking for a major chord
            transposed_chord = str(Chord('CM')+transposition)
        else:
            transposed_chord = str(Chord('cm')+transposition)
    elif by == 'Harmony':
        transpose_for = ['Chord', 'Tones']
        if str(at) == str(at).upper():
            transposition = Chord('CM').transposition_to(Chord(tonality))
        else:
            transposition = Chord('cm').transposition_to(Chord(tonality))
        mode_to_find = by == by.upper()
        transposed_chord = str(Chord(raw_chord)+transposition)
    else:
        raise KeyError("Method lookup_tonality is only implemented for by='Chord' or by='Harmony'")


    if to_find not in transpose_for:
        if to_find != 'Distance':
            return raw_row[to_find].tolist()[0]
        else:
            return (raw_row['Distance'].tolist()[0])[tonality.is_minor()]# Return tuple index 1 in minor
    else:
        transposed_row = HARMONY[HARMONY['Chord']==transposed_chord]
        if to_find != 'Distance':
            return transposed_row[to_find].tolist()[0]
        else:
            return (transposed_row['Distance'].tolist()[0])[tonality.is_minor()] # Return tuple index 1 in minor

def get_notes():
    return [Note(tone) for tone in range(1,13)]

def get_chords():
    return [Chord(tones) for tones in CHORDS_abs.values()]

def dict_sorted(dictionary, key_int, reverse=False):
    return collections.OrderedDict(sorted(dictionary.items(), key=lambda x: x[key_int], reverse=reverse))


##### BASIC CLASSES #####

# Input sample_repr can be an extraction dict (source=extractions), or a pickled Sample filename (source=pickle)
class Sample():
    def __init__(self, extractions, idee={}): #extractions, filename
        self.base = extractions
        self.idee = idee

    def __getitem__(self, note_repr):
        key = Note(note_repr)
        return self.note_freqs[key]

    def __str__(self):
        return str(self.idee)
    __repr__=__str__

    def split(self, seconds=30, n=None, hop_length=512):
        if seconds:
            # Number of frames per segment
            frames_per_seg = librosa.core.time_to_frames(seconds, sr=self.base['sr'], hop_length=hop_length, n_fft=None)

            # Number of splits
            n_segs = int(self.base['y'].shape[0] / frames_per_seg / hop_length)

            # Make resulting list of array segments
            new_samples = []

            for segment_i in range(n_segs):
                # Buid children's .base
                new_base = {}
                for extraction_key, data_object in self.base.items():
                    # For sr
                    if type(data_object) == str or type(data_object) == int:
                        new_base[extraction_key] = data_object
                    # For arrays
                    else:
                        data_object_len = data_object.shape[-1]
                        start = int(data_object_len / n_segs * segment_i)
                        finish = int(data_object_len / n_segs * (segment_i+1))
                        if extraction_key == 'y' or 'y_' in extraction_key:
                            new_base[extraction_key] = data_object[start:finish]
                        else:
                            new_base[extraction_key] = data_object[:,start:finish]

                # Build childrens .idee
                new_idee = copy.deepcopy(self.idee)
                new_idee['Snippet'] = (segment_i+1, n_segs)
                new_sample = Sample(new_base, new_idee)

                # If parent chords have been computed, children can inherit their slice of it
                if hasattr(self, 'chords'):
                    new_sample.chords = copy.deepcopy(self.chords[:,start:finish])

                # If parent's tonality info has been computed, children can record parent's info
                new_sample.parent = {}
                if hasattr(self, 'tonality'):
                    new_sample.parent['tonality'] = copy.deepcopy(self.tonality)
                if hasattr(self, 'mode'):
                    new_sample.parent['mode'] = self.mode
                if hasattr(self, 'tonality_matches'):
                    new_sample.parent['tonality_matches'] = copy.deepcopy(self.tonality_matches)
                if hasattr(self, 'tonal_clarity'):
                    new_sample.parent['tonal_clarity'] = copy.deepcopy(self.tonal_clarity)
                new_samples.append(new_sample)

            return new_samples

    def to_pickle(self, filename):
        pickle.dump(self,open(filename,'wb'))

    def build_chords(self, source='chroma'):

        # Raw tone intensity data to use
        source = self.base[source]

        # List of cords to be evaluated for distance from notes at each moment
        chords = [Chord(chord_name) for chord_name in get_chords()]
        moments = self.all_moments()
        matrix = np.zeros((len(chords), 0))

        # Notes for each moment
        for notes_now_dict in moments:
            chords_now_list = []
            for chord in chords:
                chord_match = chord.match_freqs(notes_now_dict)
                chords_now_list.append(chord_match)
            chords_now_array = np.array(chords_now_list).reshape(-1,1)
            chords_now_array = StandardScaler().fit_transform(X=chords_now_array)
            matrix = np.append(matrix, chords_now_array, axis=1)

        # Standard scale
        self.chords = matrix
        #self.chords = StandardScaler().fit_transform(matrix)

    def try_build_chords(self):
        if not hasattr(self, 'chords'):
            self.build_chords()

    # Dictionary of average chord intensities in Sample
    def chord_mean_intensities(self, sort=False):

        self.try_build_chords()
        chords = get_chords()
        tonalities = self.chords.mean(axis=1)
        tonality = collections.OrderedDict(zip(chords, tonalities))

        if sort:
            return collections.OrderedDict(sorted(tonality.items(), key=lambda x: x[1], reverse=True))
        else:
            return tonality

    def build_tonality_matches(self):

        self.try_build_chords()
        chords_dict = self.chord_mean_intensities()
        winning = Chord('CM')
        winning_score = float('-inf')

        tonality_match_dict = {}
        for tonality_name in get_chords():
            tonality = Chord(tonality_name)
            cum_sum_intensity = 0
            for chord_name, intensity in chords_dict.items():
                distance = lookup_tonality(tonality, by='Chord', at=chord_name, to_find='Distance')
                match = 5/(distance+1)
                match_gain = intensity * match
                cum_sum_intensity += match_gain

            tonality_match_dict[tonality] = cum_sum_intensity
            if cum_sum_intensity > winning_score:
                winning = tonality_name
                winning_score = cum_sum_intensity

        self.tonality_matches = tonality_match_dict
        self.tonality = Chord(winning)
        scores = sorted((list(tonality_match_dict.values())),reverse=True)
        top_score, runners_up = scores[0], [score for score in scores if score>0][1:]
        self.tonal_clarity = top_score / sum(runners_up)
        self.mode = 'Major' if self.tonality.is_major() else 'Minor'

    def try_build_tonality_matches(self):
        if not hasattr(self, 'tonality_matches'):
            self.build_tonality_matches()

    def tonalities(self, sort=False):

        if not sort:
            return self.tonality_matches
        else:
            return dict_sorted(self.tonality_matches, key_int=1, reverse=True)

    def try_build_dist_from_parent_tonality(self):
        self.try_build_tonality_matches()

        if hasattr(self, 'parent') and 'tonality' in self.parent.keys():
            cum_sum_distance = 0
            for tonality, intensity in self.tonality_matches.items():
                distance = tonality.dist_of(self.parent['tonality'])
                #dist_from_parent_tonality[tonality] = distance
                weighted_distance = distance * intensity
                cum_sum_distance += weighted_distance

            weighted_avg_distance = cum_sum_distance / len(self.tonality_matches.keys())
            self.dist_from_parent_tonality = weighted_avg_distance
            return weighted_avg_distance
        else:
            return None

    def build_harmonies():

        def chordname_to_harmony_name(tonality, chordname):
            return lookup_tonality(tonality, by='Chord', at=chordname, to_find='Harmony')

        self.try_build_chords()
        self.try_build_tonality_matches()

        harmonies = self.chords.map(chordname_to_harmony_name(self.tonality))
        # Raw tone intensity data to use
        source = self.chords

        # List of cords to be evaluated for distance from notes at each moment
        moments = self.all_moments()
        matrix = np.zeros((len(HARMONY), 0))

        # Notes for each moment
        for notes_now_dict in moments:
            chords_now_list = []
            for chord in chords:
                chord_match = chord.match_freqs(notes_now_dict)
                chords_now_list.append(chord_match)
            chords_now_array = np.array(chords_now_list).reshape(-1,1)
            chords_now_array = StandardScaler().fit_transform(X=chords_now_array)
            matrix = np.append(matrix, chords_now_array, axis=1)

        self.chords = matrix
        #self.chords = StandardScaler().fit_transform(matrix) ??

    def note_freqs(self, sort=False):
        freqs_dict = collections.OrderedDict()
        for note in get_notes():
            freqs_dict[note] = note.freqs_in(self)
        if sort: freqs_dict = sorted(freqs_dict.items(), key=lambda t: t[1], reverse=True)
        return freqs_dict

    def chord_freqs(self, rootweight=3, sort=False):
        freqs_dict = collections.OrderedDict()
        for chord in get_chords():
            freqs_dict[chord] = chord.freqs_in(self, rootweight)
        return freqs_dict

    def mean_note_freqs(self, sort=False):
        meanfreqs_dict = collections.OrderedDict()
        for note in get_notes():
            meanfreqs_dict[note] = note.mean_freq_in(self)
        if sort: meanfreqs_dict = sorted(meanfreqs_dict.items(), key=lambda t: t[1], reverse=True)
        return meanfreqs_dict

    def mean_chord_freqs(self, sort=False, rootweight=3):
        meanfreqs_dict = collections.OrderedDict()
        for chord in get_chords():
            meanfreqs_dict[chord] = chord.mean_freq_in(self, rootweight=rootweight)
        if sort: meanfreqs_dict = sorted(meanfreqs_dict.items(), key=lambda t: t[1], reverse=True)
        return meanfreqs_dict

    def moment(self, i):
        moment = self.base['chroma'][:,i]
        moment_dict = {}
        for note in get_notes():
            moment_dict[note] = moment[note.tone-1]
        return moment_dict

    # Returns list of all the momentary frequencies in Sample
        # Functions like a pivot on self.notes_freqs,
        # transforming it from a note_dict of arrays to a list of note_dicts
    def all_moments(self):
        return [self.moment(i) for i in range(self.base['chroma'].shape[1])]

    # Takes a moment -- a dict of note frequencies at a particular moment
    # Returns the distance for each chord to that moment -- a {Chord_i:dist_float_i...}
    def chords_dist_at(self, moment):
        chords_dist_dict = collections.OrderedDict()
        chords = get_chords()
        for chord in chords:
            chords_dist_dict[chord] = chord.dist_of_freqs(moment)
        return chords_dist_dict

    # Represents intervalic intensity at each moment of Sample
    # Row 1 is half-steps (i.e. m2 and M7), row 2 is whole_steps(i.e. M2 and m7), etc
    # Row 0 is unison, and it is interpreted extremely literally as always equal to 0
    def interval_matrix(self):

        # What it sounds like
        def mean(freq1, freq2):
            return (freq1+freq2)/2

        # Average of freqs between of tone_i index and the tone int_above steps above it
        def int_freq(tone_i, int_above, freqs):
            int_tone_i = (tone_i+int_above)%12
            return mean(freqs[tone_i], freqs[int_tone_i])

        # Array of average freqs between tone_i and the next 6 steps above it
        def tone_int_freqs(tone_i, freqs):
            int_freqs = np.zeros(7)
            for interval in range(1,7):
                int_freqs[interval] = int_freq(tone_i, interval, freqs)
            return int_freqs

        # Sum of the above, across each tone
        def sum_int_freqs(freqs):
            int_freqs = np.zeros(7)
            for tone_i in range(0,11):
                int_freqs += tone_int_freqs(tone_i, freqs)
            return int_freqs

        # Construct interval matrix by iterating across each moment
        int_freqs_list = []
        moments = self.base['chroma'].T

        for moment in moments:
            moment_int_freqs = sum_int_freqs(moment)
            int_freqs_list.append(moment_int_freqs)

        int_matrix = np.array(int_freqs_list).T
        return int_matrix

    def interval_summary(self):
        return np.mean(self.interval_matrix(),axis=1)

    def display_intervals(self, bins=20):
        im = self.interval_matrix()
        display = im[1:-1,::im.shape[1]//bins] # Skip TT for display purposes
        sns.set(font_scale = 3)
        fig, ax = plt.subplots(figsize=(20,10))
        sns.heatmap(display)
        ax.set(xlabel='Time', ylabel='Interval')
        ax.invert_yaxis()
        ax.set(yticklabels=[1,2,3,4,5])

    def interval_class_summary(self):
        return np.mean(self.interval_class_matrix(), axis=1)

    def interval_class_matrix(self):

        # Collect and scale an interval_matrix across song
        im = self.interval_matrix()
        scaler = StandardScaler()
        scaler.fit(im)
        im_scaled = scaler.transform(im)
        im_scaled

        # Reduce to dissonace, consonance, and perfect intervals
        D = im[1]+im[6]
        C = im[3]+im[4]
        P = im[5]
        DCP = np.array([D,C,P])
        return DCP

    def display_interval_classes(self, bins=20):
        #hop_length=512
        DCP = self.interval_class_matrix()
        display = DCP[:,::DCP.shape[1]//bins]
        sns.set(font_scale = 3)
        fig, ax = plt.subplots(figsize=(20,10))
        sns.heatmap(display)
        ax.set(xlabel='Time', ylabel='Interval Class')
        ax.invert_yaxis()
        ax.set(yticklabels=['Dissonance','Consonance','Perfect'])

    def polyphony_array(self):
        """
        Estimates the level of polyphony across Sample
        """
        # Variance between pitch-class intensity at each moment in time
        return np.var(self.base['chroma'], axis=0)

    def mean_polyphony(self):
        return np.mean(self.polyphony_array())

    def var_polyphony(self):
        return np.var(self.polyphony_array())

    def display_polyphony(self, bins=20):
        polyphony = self.polyphony_array()

        from scipy.interpolate import spline
        xnew = np.linspace(polyphony.min(),polyphony.max(),100) #300 represents number of points to make between T.min and T.max
        power_smooth = spline(polyphony,power,xnew)

        #hop_length=512
        display = polyphony[xnew, power_smooth]
        sns.set(font_scale = 3)
        fig, ax = plt.subplots(figsize=(20,10))
        sns.heatmap(display)
        ax.set(xlabel='Time', ylabel='Polyphony')
        ax.invert_yaxis()
        #ax.set(yticklabels=['Dissonance','Consonance','Perfect'])

    def display_chords(self, bins=30, exaggerate=False):
        if not hasattr(self, 'chords'):
            self.build_chords()

        #hop_length=512
        chord_names = list(str(get_chords()))
        if exaggerate:
            if exaggerate%2:
                display_chords = self.chords**exaggerate
            else:
                display_chords = self.chords**(exaggerate+1)
        else:
            display_chords = self.chords


        #times = librosa.frames_to_time(display_chords)
        #times = np.array(times)

        #print(times)
        #plt.figure(figsize=(18, 8))
        #sns.heatmap(display_chords)
        #plt.xlabel('Time')
        #plt.ylabel('Chord')
        #ax = plt.gca()
        #ax.invert_yaxis()
        #ax.set_yticklabels(labels=all_chords, rotation=0, fontsize=10)
        #ax.set_xticklabels(labels=times, rotation=0, fontsize=10)
        #plt.axis('tight')
        #plt.tight_layout()
        #plt.title('Chord Match');


        times = librosa.frames_to_time(display_chords)
        times = np.array(times)

        chords = [str(chord) for chord in get_chords()]
        #natural_chords = [chord for chord in chords if '#' not in chord]

        sns.set(font_scale = 1)

        #display = display_chords[:,::len(display_chords)//bins]
        display = display_chords
        fig, ax = plt.subplots(figsize=(18,8))

        sns.heatmap(display)
        ax.set(xlabel='Frame', ylabel='Chord')
        ax.invert_yaxis()
        #ax.set(yticklabels=chord_names)
        ax.set_yticklabels(labels=chords, rotation=0)
        #ax.set(xticklabels=times)
        ax.set_title('Chord Match')

# Constructs a note from a note representation
# such as a tone (int 1-12), a note name (valid natural or sharp), or another Note
class Note():
    def __init__(self, note_repr, intensity=None):
        if type(note_repr) == Note:
            self.tone = note_repr.tone
            self.name = note_repr.name
        elif type(note_repr) == int or note_repr == type(note_repr) == float:# In case of an equivalent float
            note_repr = int(note_repr)
            self.tone = reduce_oct(note_repr)
            self.name = TONES_TO_STR[reduce_oct(note_repr)]
        elif type(note_repr) == str:
            self.tone = STRS_TO_TONE[note_repr.upper()]
            self.name = note_repr.upper()
        else:
            raise TypeError('Note class constructor can only take a tone (int 0-12) or a valid note name (str, natural or sharp -- i.e. C, D#, f, etc).')

        if intensity:
            self.intensity = intensity

    def __str__(self):
        return str(self.name)
    __repr__ = __str__

    def __cmp__(self, other):
        return cmp(self.tone, other.tone)

    def __lt__(self, other):
        return self.tone < other.tone

    def __gt__(self, other):
        return self.tone > other.tone

    def __add__(self, halfsteps):
        return Note(self.tone + halfsteps)

    def __sub__(self, halfsteps):
        return Note(self.tone - halfsteps)

    def __iadd__(self, halfsteps):
        return Note(self.tone + halfsteps)

    def __isub__(self, halfsteps):
        return Note(self.tone - halfsteps)

    def __hash__(self):
        return hash(self.tone)

    def __eq__(self, other):
        return (self.tone) == (other.tone)

    def copy(self):
        new_copy = Note(self.tone)
        return new_copy

    def freqs_in(self, sample):
        return sample[self]

    def mean_freq_in(self, sample):
        return np.mean(self.freqs_in(sample))

    # Harmonic distance to another note (Linear)
    def dist_of(self, other):
        transposition = (self-1).tone
        transposed_other = (other-transposition)
        transposed_other_tone = transposed_other.tone
        distance = DIST_OF_TONE[transposed_other_tone]
        return distance

class Chord():

    # Create a chord from a chord representation, such as a tuple of notes, tones, a chord name, or a Chord
    def __init__(self, chord_repr):
        if type(chord_repr) == tuple or type(chord_repr) == list:
            self.notes = tuple(Note(note_repr) for note_repr in chord_repr)
        elif type(chord_repr) == Chord:
            self.notes = chord_repr.notes
        elif type(chord_repr) == str:
            self.notes = tuple(Note(note_repr) for note_repr in CANONICAL_CHORD_TONES[chord_repr])
        else:
            raise TypeError(str(type(chord_repr)), ': Chord constructor can only take tuples of notes, tuples of Notes, a chord name, or Chord object.')
        self.name = CANONICAL_TONES_TO_CHORD[tuple(sorted(note.tone for note in self.notes))]
        self.tones = tuple([note.tone for note in self.notes])
        self.root = (self.name.replace('M','').replace('m','')).upper()
        self.roottone = STRS_TO_TONE[self.root]
        self.intensity = None

    def __str__(self):
        return str(self.name)
    __repr__ = __str__

    def __add__(self, halfsteps):
        return Chord([note.tone + halfsteps for note in self.notes])

    def __sub__(self, halfsteps):
        return Chord([note.tone - halfsteps for note in self.notes])

    def __iadd__(self, halfsteps):
        return Chord([note.tone + halfsteps for note in self.notes])

    def __isub__(self, halfsteps):
        return Chord([note.tone - halfsteps for note in self.notes])

    def __eq__(self, other):
        return self.notes == other.notes

    def __hash__(self):
        return hash(self.notes)

    def major(self):
        return Chord(self.name.upper())

    def minor(self):
        return Chord(self.name.lower())

    def is_major(self):
        return self.name == self.name.upper()

    def is_minor(self):
        return self.name == self.name.lower()

    def transposition_to(self, other_chord):
        return reduce_oct(other_chord.roottone - self.roottone)

    def freqs_in(self, sample, rootweight=1):
        weighted_sums_array = sample[self.root] * rootweight
        for note in self.notes[1:]:
            weighted_sums_array += sample[note]
        divisor = rootweight + len(self.notes) - 1
        return weighted_sums_array / divisor

    def mean_freq_in(self, sample, rootweight=1):
        return np.mean(self.freqs_in(sample, rootweight=rootweight))

    # Harmonic distance to another chord (Linear)
    # Defined as average of shortest note distances to other's notes
    def dist_of(self, other):
        sum = 0
        divisor = len(self.notes)
        for self_note in self.notes:
            shortest_distance = 6
            for other_note in other.notes:
                distance = self_note.dist_of(other_note)
                if (distance < shortest_distance): shortest_distance = distance
            sum += shortest_distance
        return sum/divisor

    # Harmonic distance to a group of tones (Linear)
    # Defined as average of shortest note distances to other's tones
    def dist_of_notes(self, notes):
        sum = 0
        divisor = len(self.notes)
        for other_note in notes:
            shortest_distance = 6
            for self_note in self.notes:
                distance = self_note.dist_of(other_note)
                if (distance < shortest_distance): shortest_distance = distance
            sum += shortest_distance
        return sum/divisor

########

    # Return the harmonic distance between another note and the chord
    def dist_of_note(self, other_note):
        shortest_distance = 6
        for self_note in self.notes:
            distance = self_note.dist_of(other_note)
            if (distance < shortest_distance): shortest_distance = distance
        return shortest_distance

    # Return the harmonic distance between the chord and another note weighted by otherfreq
    def dist_of_freq(self, other_note, other_notefreq):
        return self.dist_of_note(other_note) * other_notefreq

    # Return the average harmonic distance between Chord and a group of tones, weighted by their freqs
    def dist_of_freqs(self, other_notes_freq_dict):
        sum = 0
        divisor = len(self.notes)
        for other_note, other_notefreq in other_notes_freq_dict.items():
            sum += self.dist_of_freq(other_note, other_notefreq)
        return sum / divisor

    # How close a match to a dict of tone intensities
    def match_freqs(self, other_notes_freq_dict):
        return 1 / (self.dist_of_freqs(other_notes_freq_dict)+1)
#######

# Pass chromslice, or a correctly formatted df, if available
# Assumes in root position
    def freq(self, noteFreq_df=None, chromslice=None):
        if noteFreq_df is None and chromslice is None:
            raise Error ('Chord.strength requires either a note_freq_df or chromslice')
        if noteFreq_df is None:
            noteFreq_df = note_freq_df(chromslice)
        if noteFreq_df.index.name != 'Note':
            noteFreq_df = noteFreq_df.set_index('Note')
        noteFreq_df.sort_index(ascending=False, inplace=True)

        weighted_sum = noteFreq_df.iloc[self.notes[0]]['Note_Freq']

        return weighted_sum

######################
###### ANALYSIS ######
######################

def tone_freq(chromslice):
    tones = np.arange(1,13,1)
    tone_freq_array = chromslice.mean(axis=1).tolist()
    return sorted([tone for tone in zip(tones, tone_freq_array)], reverse=True, key=lambda x: x[1])

def note_freq_dict(chromslice):
    tone_freq_array = tone_freq(chromslice)
    notes = [Note(tone[0]) for tone in tone_freq_array]
    nf = [tone[1] for tone in tone_freq_array]

    freq_dict = OrderedDict()

def note_freq_df(chromslice):
    tone_freq_array = tone_freq(chromslice)
    notes = [Note(tone[0]) for tone in tone_freq_array]
    nf = [tone[1] for tone in tone_freq_array]

    df = pd.DataFrame()
    df['Note'] = notes
    df['Note_Freq'] = nf

    df.sort_values('Note_Freq',ascending=False)
    df = df.set_index('Note')
    return df

def distance(chord1, chord2):
    tones1 = canonized(chord1.tones)
    tones2 = canonized(chord2.tones)

    difference = []
    for note in zip(tones1, tones2):
        difference.append(tones2-tones1+1)
    difference = tuple(abs(canonized(difference)))
    harmony2 = HARMONIES[difference]

    if chord1.is_major():
        return MAJ_DIST[harmony2]
    else:
        return MIN_DIST[harmony2]

#####################
###### VISUALS ######
#####################

def display_chroma(chromagram, sr=22050, subplot_num=1, title='Chromagram'):
# Adapted from https://librosa.github.io/librosa/tutorial.html
    plt.figure(figsize=(20,20))
    sns.set(font_scale = 3)
    plot = plt.subplot(2, 1, subplot_num)
    librosa.display.specshow(chromagram, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    return plot

######################################
###### SAVE/LOAD PICKLE CONTENT ######
######################################

# Holding fast to Pandas' idiosyncratic convention of asymmertrically naming functions to_pickle and read_pickle
def to_pickle(base_filename, extractions_dict, extension='.pkl'):
    for key in extractions_dict.keys():
        data = extractions_dict[key]
        filename = base_filename.split('.')[0]
        filename += '_' + key
        filename += extension
        pickle.dump(data,open(filename,'wb'))

def try_get_pickle(filename):
    try:
        return get_pickle(filename)
    except:
        return None

def get_pickle(filename):
    return pickle.load(open(filename,'rb'))

def read_pickle(base_filename, extraction_keys=['y','sr','y_harmonic','y_percussive','chroma','sparse'], extension='.pkl'):
    extractions = {}
    for key in extraction_keys:
        filename = base_filename.split('.')[0]
        filename += '_' + key
        filename += extension
        datum = pickle.load(open(filename,'rb'))
        extractions[key] = datum
    return extractions

def parse_audio(audio_path, extraction_keys=['y','sr','y_harmonic','y_percussive','chroma','sparse']):
    extractions = {}
    sr = 22050

    # Load parse audio with librosa
    y, sr = librosa.load(audio_path)
    extractions['y'] = y
    extractions['sr'] = sr

    # Get harmonics and percussives
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    extractions['y_harmonic'] = y_harmonic
    extractions['y_percussive'] = y_percussive

    # Collect harmonic component
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    extractions['chroma'] = chroma

    # Sparse chromagram
    if 'sparse' in extraction_keys:
        sparse = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=12*6)
        extractions['sparse'] = sparse

    return extractions

def parse_and_pickle(audio_path):
    extractions = parse_audio(audio_path)
    to_pickle(audio_path, extractions)
    return extractions

def load_or_parse(audio_path, extraction_keys=['y','sr','y_harmonic','y_percussive','chroma','sparse'], auto_pickle=True):
    try:
        parse_dict = read_pickle(audio_path, extraction_keys)
        return parse_dict
    except IOError:
        if auto_pickle:
            print('Failed to locate extracted data for',audio_path.split('/')[-1],'-- extracting, pickling, and returning.')
            return parse_and_pickle(audio_path)
        else:
            return parse_audio(audio_path)
