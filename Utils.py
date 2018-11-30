""" Implements constants and helper functions for MTheory """

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

# For tones > 12, drop their octave so that tone becomes same note in range [0-12]
def reduce_oct(tone_or_tones):
    """ Canonizes tone values > 12 to the standard 0-11 range """
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
    """ Transposes a tone by half-steps.
        tone: int
        by: +/- int half-steps to transpose
    """
    return reduce_oct(tone + by)

def transpose_tones(tones, by):
    """ Returns a tuple of transposed tones by half-steps.
        tones: iterable of int tones
        by: +/- int half-steps to transpose by
    """
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
STRS_TO_TONE = dict([(note, tone) for tone, note in TONES_TO_STR])

# Canonical chord tones
# TODO: diminished and 7th chords
CHORD_TYPES = {'maj'      :(1,5,8),    # Major
               'min'      :(1,4,8)     # Minor
               #'dim'      :(1,4,7)    # Diminished
               #'dim7':(1,4,7,11)
               #'maj7'     :(1,5,8,11)
}

def _defineCHORDS_abs():
    """ Defines major and minor chords by their tones.
        i.e. {'C':(1,5,8), 'c':(1,4,8)...}
    """
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

def _defineCANONICAL_CHORD_TONES():
    """ Defines canonical chords based on sorting their tones.
        i.e {'C':(1,5,8)...'g':(3,7,12)}
            {(1,5,8):'C'...(3,7,12):'g'}
    """
    CANONICAL_CHORD_TONES = {}
    CANONICAL_TONES_TO_CHORD = {}
    for chord, tones in CHORDS_abs.items():
        canonical_tones = tuple(sorted(tones))
        CANONICAL_CHORD_TONES[chord] = canonical_tones
        CANONICAL_TONES_TO_CHORD[canonical_tones] = chord
    return CANONICAL_CHORD_TONES, CANONICAL_TONES_TO_CHORD
CANONICAL_CHORD_TONES, CANONICAL_TONES_TO_CHORD = _defineCANONICAL_CHORD_TONES()

""" Canonical circle-of-fifths of distances from C. """
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

""" Harmony definitions:
    Tones: Harmonies to tones, canonized to CM
    Distance: Distance function lookup. Tuple distances to tonal (<major>, <minor>)
              for common-practice period. (Derivation: circle-of-fifths and domain expertise.)
"""
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
    """ Uses a pandas DataFrame as a lookup table to convert between
        Chords, Harmonies, Tones, and Distances.
    """
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
    """ Get all 11 notes """
    return [Note(tone) for tone in range(1,13)]

def get_chords():
    """ Get all chords """
    return [Chord(tones) for tones in CHORDS_abs.values()]

def dict_sorted(dictionary, key_int, reverse=False):
    """ Returns a sorted OrderedDict from a dict """
    return collections.OrderedDict(sorted(dictionary.items(), key=lambda x: x[key_int], reverse=reverse))

########################
########### IO #########
########################

def to_pickle(self, filename):
    pickle.dump(self,open(filename,'wb'))
