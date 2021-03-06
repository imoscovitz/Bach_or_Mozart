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
from MTheoryUtils import *

class Sample():
    """ Class to manage an audio clip.

        Important attributes:

        .base: dict of basic Librosa-extracted audio data.
               Typical Sample.base will contain 'y','sr','y_harmonic','y_percussive','chroma','sparse'
        .idee: Sample identifier

        """

    def __init__(self, extractions, idee={}):
        self.base = extractions
        self.idee = idee

    def __getitem__(self, note_repr):
        key = Note(note_repr)
        return self.note_freqs[key]

    def __str__(self):
        return str(self.idee)
    __repr__=__str__

    def split(self, seconds=30, n=None, hop_length=512):
        """ Splits a Sample into evenly-sized chronological pieces. """
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

    def build_chords(self, source='chroma'):
        """ Based on notes, computes and stores (in Sample.chords attribute)
            a matrix representing the respective strengths of each chord
            at each moment in time.
        """
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
        """ Builds Sample chords if they haven't been built already. """
        if not hasattr(self, 'chords'):
            self.build_chords()

    # Dictionary of average chord intensities in Sample
    def chord_mean_intensities(self, sort=False):
        """ Produces a dictionary of each chord's mean intensity
            over the chronological length of Sample.
        """
        self.try_build_chords()
        chords = get_chords()
        tonalities = self.chords.mean(axis=1)
        tonality = collections.OrderedDict(zip(chords, tonalities))

        if sort:
            return collections.OrderedDict(sorted(tonality.items(), key=lambda x: x[1], reverse=True))
        else:
            return tonality

    def build_tonality_matches(self):
        """ Based on a sample's computed Chords, computes the closeness of fit
            for each tonality, the best-fitting tonality, and a "tonal purity"
            score for the Sample.
        """
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
        """ Builds tonality matches if they haven't been built already. """
        if not hasattr(self, 'tonality_matches'):
            self.build_tonality_matches()

    def tonalities(self, sort=False):
        """ Return the tonalities of a Sample (with the option to sort) """
        if not sort:
            return self.tonality_matches
        else:
            return dict_sorted(self.tonality_matches, key_int=1, reverse=True)

    def try_build_dist_from_parent_tonality(self):
        """ If a Sample represents just a portion of a longer piece of audio,
            and its parent's tonality is known, calculate the tonal remoteness
            of the Sample from its parent Sample.
        """
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
        """ Estimate the harmonies at each moment of the Sample """
        def chordname_to_harmony_name(tonality, chordname):
            return lookup_tonality(tonality, by='Chord', at=chordname, to_find='Harmony')

        self.try_build_chords()
        self.try_build_tonality_matches()

        harmonies = self.chords.map(chordname_to_harmony_name(self.tonality))
        # Raw tone intensity data to use
        source = self.chords

        # List of chords to be evaluated for distance from notes at each moment
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

    """ Sample class helpers: """

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

    def all_moments(self):
        """ Returns list of all the momentary frequencies in Sample

            Useful for pivoting on self.notes_freqs,
            transforming it from a note_dict of arrays to a list of note_dicts
        """
        return [self.moment(i) for i in range(self.base['chroma'].shape[1])]


    def chords_dist_at(self, moment):
        """ Input: dict of note frequencies at a particular moment
            Returns the distance for each chord to that moment -- a {Chord_i:dist_float_i...}
        """
        chords_dist_dict = collections.OrderedDict()
        chords = get_chords()
        for chord in chords:
            chords_dist_dict[chord] = chord.dist_of_freqs(moment)
        return chords_dist_dict

    def interval_matrix(self):
        """ Returns a matrix representing the intensity of each chromatic intervals
            over the chronological length of the sample.

            Row 1 is half-steps (i.e. m2 and M7), row 2 is whole_steps(i.e. M2 and m7), etc
            Row 0 is unison, and it is interpreted extremely literally as always equal to 0
        """

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
        """ Returns the average intensity of each interval over the Sample's
            chronological length
        """
        return np.mean(self.interval_matrix(),axis=1)

    def display_intervals(self, bins=20, show_TT=False):
        """ Produces a matplotlib visualization of intervalic intensity """

        im = self.interval_matrix()
        if not show_TT: # Skip TT for display purposes b/c they're so rare
            display = im[1:-1,::im.shape[1]//bins]
        else:
            display = im[1:,::im.shape[1]//bins]
        sns.set(font_scale = 3)
        fig, ax = plt.subplots(figsize=(20,10))
        sns.heatmap(display)
        ax.set(xlabel='Time', ylabel='Interval')
        ax.invert_yaxis()
        ax.set(yticklabels=[1,2,3,4,5])

    def interval_class_summary(self):
        """ Returns the average intensity of interval classes (dissonant, consonant, perfect) """
        return np.mean(self.interval_class_matrix(), axis=1)

    def interval_class_matrix(self):
        """ Returns a matrix representing intensity over the course of the Sample
            of interval groups dissonant, consonant, and perfect.
        """
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
        """ Produces matplotlib visualization of interval classes """
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
        """ Estimates the level of polyphony across Sample. """
        return np.var(self.base['chroma'], axis=0)

    def mean_polyphony(self):
        """ Estimates the average polyphony over the entire Sample. """
        return np.mean(self.polyphony_array())

    def var_polyphony(self):
        """ Estimates the variance of polyphony over the entire Sample. """
        return np.var(self.polyphony_array())

    def display_polyphony(self, bins=20):
        """ Produces matplotlib visualization of polyphony over the Sample. """
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
        """ Produces matplotlib visualization of computed chord intensities over the Sample """
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

class Note():
    """ Class representing a note. Underlying each Note is an int tone 0-11. """

    def __init__(self, note_repr, intensity=None):
        """ Construct a Note from another Note, a tone, or a string representation of a Note """
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

    def dist_of(self, other):
        """ Harmonic distance to another note.
            Measured in linear distance, derived from circle-of-fifths.
        """
        transposition = (self-1).tone
        transposed_other = (other-transposition)
        transposed_other_tone = transposed_other.tone
        distance = DIST_OF_TONE[transposed_other_tone]
        return distance

class Chord():
    """ Class representing a chord, i.e. a group of notes. """

    def __init__(self, chord_repr):
        """ Construct a chord from a chord representation,
            such as a tuple of notes, tones, a chord name, or another Chord.
        """

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

    def dist_of(self, other):
        """ Harmonic distance to another chord.
            Defined as the average of the shortest note-pair distances to the other Chord's notes.
        """
        sum = 0
        divisor = len(self.notes)
        for self_note in self.notes:
            shortest_distance = 6
            for other_note in other.notes:
                distance = self_note.dist_of(other_note)
                if (distance < shortest_distance): shortest_distance = distance
            sum += shortest_distance
        return sum/divisor

############################
##### DISTANCE HELPERS #####
############################

    def dist_of_note(self, other_note):
        """ Returns the harmonic distance between another note and the chord. """
        shortest_distance = 6
        for self_note in self.notes:
            distance = self_note.dist_of(other_note)
            if (distance < shortest_distance): shortest_distance = distance
        return shortest_distance

    def dist_of_freq(self, other_note, other_notefreq):
        """ Return the harmonic distance between the chord and another note weighted by otherfreq. """
        return self.dist_of_note(other_note) * other_notefreq

    def dist_of_freqs(self, other_notes_freq_dict):
        """ Return the average harmonic distance between Chord and a group of tones,
            weighted by their intensities.
        """
        sum = 0
        divisor = len(self.notes)
        for other_note, other_notefreq in other_notes_freq_dict.items():
            sum += self.dist_of_freq(other_note, other_notefreq)
        return sum / divisor


    def match_freqs(self, other_notes_freq_dict):
        """ Returns how close a match is to a dict of tone intensities. """
        return 1 / (self.dist_of_freqs(other_notes_freq_dict)+1)

    def freq(self, noteFreq_df=None, chromslice=None):
        """ Takes chromslice, or a correctly formatted df, if available.
            Assumes chord is in root position.
        """
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
    """ Produces matplotlib visualization of a chromograph.
        Modified from https://librosa.github.io/librosa/tutorial.html
    """
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

def to_pickle(base_filename, extractions_dict, extension='.pkl'):
    """ Pickle a dictionary of librosa extractions.

        Holding fast to Pandas' idiosyncratic convention
        of asymmertrically naming functions "to_pickle" and "read_pickle"
    """
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
    """ Extract relevant data from raw .mp3 file. """

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
