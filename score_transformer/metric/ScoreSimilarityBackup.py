from enum import IntEnum
import music21
import numpy as np

class ScoreErrors(IntEnum):
    Clef = 0
    KeySignature = 1
    TimeSignature = 2
    NoteDeletion = 3
    NoteInsertion = 4
    NoteSpelling = 5
    NoteDuration = 6
    StemDirection = 7
    Beams = 8 # added
    Tie = 9 # added
    RestInsertion = 10
    RestDeletion = 11
    RestDuration = 12
    StaffAssignment = 13
    Voice = 14 # added

def score_alignment(aScore, bScore):
    """Compare two musical scores.

    Parameters:

    aScore/bScore: music21.stream.Score objects

    Return value:

    (path, d):
           path is a list of tuples containing pairs of matching offsets
           d is the alignment matrix
    """
    def convertScoreToListOfPitches(aScore):
        """Convert a piano score into a list of tuples containing pitches

        Parameter:
            aScore a music21.Stream containing two music21.stream.PartStaff

        Return value:
            list of tuples (offset, pitches)
                offset is a real number indicating the offset of an object in music21 terms
                pitches is a list of pitches in MIDI numbers
        """
        my_dict = {}
        for n in aScore.flatten().notes:
            if n.offset not in my_dict:
                my_dict[n.offset] = set()
            if n.isChord:
                for sn in n:
                    my_dict[n.offset].add(sn.pitch.midi)
            else:
                my_dict[n.offset].add(n.pitch.midi)
        offsets = list(my_dict.keys())
        assert sorted(offsets) == offsets
        pitches = list(my_dict.values())
        # [(k, v) for k, v in my_dict.items()]
        return offsets, pitches

    def costMatrix(s, t):
        d = np.zeros((len(s) + 1, len(t) + 1))
        d[1:, 0] = np.inf
        d[0, 1:] = np.inf
        for j, t_ in enumerate(t):
            for i, s_ in enumerate(s):
                d[i + 1, j + 1] = min(d[i, j + 1], d[i + 1, j], d[i, j]) + len(t_ ^ s_)
        return d

    # score_alignment
    offsets_a, pitches_a = convertScoreToListOfPitches(aScore)
    offsets_b, pitches_b = convertScoreToListOfPitches(bScore)

    d = costMatrix(pitches_a, pitches_b)

    i, j = (d.shape[0] - 1, d.shape[1] - 1)
    path = []
    while i and j:
        aOff = offsets_a[i-1]
        bOff = offsets_b[j-1]
        path = [(aOff, bOff)] + path

        idx = np.argmin([d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]])
        if idx == 0:
            i = i - 1
        elif idx == 1:
            j = j - 1
        else:
            i, j = i - 1, j - 1

    return path, d


def score_similarity(estScore, gtScore, partMapping={0: "right", 1: "left"}):
    """Compare two musical scores.

    Parameters:

    estScore/gtScore: music21.stream.Score objects of piano scores. The scores must contain two
        music21.stream.PartStaff substreams (top and bottom staves)

    estScore is the estimated transcription
    gtScore is the ground truth

    Return value:

    A NumPy array containing the differences between the two scores:

        barlines, clefs, key signatures, time signatures, note, note spelling,
        note duration, staff assignment, rest, rest duration

    The differences for notes, rests and barlines are normalized with the number of symbols
    in the ground truth.
    """

    if isinstance(estScore, str):
        estScore = music21.converter.parse(estScore).expandRepeats()
    if isinstance(gtScore, str):
        gtScore = music21.converter.parse(gtScore).expandRepeats()
    assert isinstance(estScore, music21.stream.Stream)
    assert isinstance(gtScore, music21.stream.Stream)

    def countSymbols(aScore):
        """Count the number of symbols in a score

        Parameter:
            aScore a music21.Stream

        Return value:
            the number of music symbols (notes, rests, chords, barlines) in the score
        """

        # Classes to consider
        CLASSES = [music21.note.Note, music21.chord.Chord, music21.note.Rest]
        nSymbols = {'n_' + cls.__name__: sum([len(el.notes) if cls == music21.chord.Chord else 1
                                    for el in aScore.recurse() if isinstance(el, cls)])
                    for cls in CLASSES}
        return nSymbols

    def convertScoreToList(aScore):
        """Convert a piano score into a list of tuples

        Parameter:
            aScore a music21.Stream containing two music21.stream.PartStaff

        Return value:
            list of tuples (offset, staff, object)
                offset is a real number indicating the offset of an object in music21 terms
                staff is an integer indicating the staff (0 = top, 1 = bottom)
                object is a music21 object
        """

        # Classes to consider
        CLASSES = (music21.bar.Barline, music21.note.Note, music21.note.Rest, music21.chord.Chord)

        def convertStreamToList(s, hand):
            elements_with_offsets = {}
            for el in s.recurse():
                if isinstance(el, CLASSES):
                    offset = el.getOffsetInHierarchy(s)
                    elements_with_offsets.setdefault(offset, []).append((hand, el))
            return sorted([(k, v) for k, v in elements_with_offsets.items()])

        def flattenStream(s):
            new_stream = music21.stream.Stream()
            for el in s.flatten():
                if isinstance(el, (music21.note.Note, music21.note.Rest, music21.chord.Chord)):
                    new_stream.insert(el)
            for el in s.recurse():
                if isinstance(el, (music21.stream.Measure, music21.bar.Barline)):
                    new_stream.insert(el.getOffsetInHierarchy(s), music21.bar.Barline())
            return new_stream

        parts = aScore.getElementsByClass([music21.stream.PartStaff, music21.stream.Part])  # get staves
        topStaffList = []
        bottomStaffList = []
        for i, part in enumerate(parts):
            if partMapping[i] == "right":
                topStaffList += convertStreamToList(flattenStream(part), 0)
            else:
                bottomStaffList += convertStreamToList(flattenStream(part), 1)

        aList = []
        tIterator = iter(topStaffList)
        bIterator = iter(bottomStaffList)
        tEl = next(tIterator, None)
        bEl = next(bIterator, None)

        while tEl or bEl:
            if not tEl:
                aList.append((bEl[0], bEl[1]))
                bEl = next(bIterator, None)
            elif not bEl:
                aList.append((tEl[0], tEl[1]))
                tEl = next(tIterator, None)
            else:
                if tEl[0] < bEl[0]:
                    aList.append((tEl[0], tEl[1]))
                    tEl = next(tIterator, None)
                elif tEl[0] > bEl[0]:
                    aList.append((bEl[0], bEl[1]))
                    bEl = next(bIterator, None)
                else:
                    aList.append((tEl[0], tEl[1] + bEl[1]))
                    tEl = next(tIterator, None)
                    bEl = next(bIterator, None)
        print(aList[:10])

        return aList

    def countObjects(a_set):
        """Count objects in a set

        Parameters:

        a_set: list of tuples (staff, object)
            staff is an integer indicating the staff (1 = top, 2 = bottom)
            object is a music21 object

        Return value:

            a tuple with the numbers of objects in the set (see definition of errors below)
        """

        errors = np.zeros((len(ScoreErrors.__members__)), int)

        for obj in a_set:
            if isinstance(obj[1], music21.note.Note):
                errors[ScoreErrors.NoteDeletion] += 1
            elif isinstance(obj[1], music21.chord.Chord):
                errors[ScoreErrors.NoteDeletion] += len(obj[1].pitches)
            elif isinstance(obj[1], music21.note.Rest):
                errors[ScoreErrors.RestDeletion] += 1
        #     if isinstance(obj[1], (music21.stream.Measure, music21.bar.Barline, music21.clef.Clef, \
        #                             music21.key.Key, music21.key.KeySignature, music21.meter.TimeSignature)):
        #         pass
            # else:
            #     print('Class not found:', type(obj[1]))

        return errors

    def compareSets(a_set, b_set):
        """Compare two sets of concurrent musical objects.

        Parameters:

        a_set/b_set: list of tuples (staff, object)
            staff is an integer indicating the staff (1 = top, 2 = bottom)
            object is a music21 object

        Return value:

            A tuple with the differences between the two sets (see definition of errors below)
        """

        def findEnharmonicEquivalent(note, a_set):
            """Find the first enharmonic equivalent in a set

            Parameters:

            note: a music21.note.Note object
            a_set: list of tuples (staff, object)
                staff is an integer indicating the staff (0 = top, 1 = bottom)
                object is a music21 object

            Return value:

                index of the first enharmonic equivalent of note in a_set
                -1 otherwise
            """
            for i, obj in enumerate(a_set):
                if isinstance(obj[1], music21.note.Note) and obj[1].pitch.ps == note.pitch.ps:
                    return i
            return -1

        def splitChords(a_set):
            """Split chords into seperate notes

            Parameters:

            a_set: list of tuples (staff, object)
                staff is an integer indicating the staff (0 = top, 1 = bottom)
                object is a music21 object

            Return value:
                a tuple (newSet, chords)
                newSet: a_set with split chords
                chords: the number of chords in a_set

            """
            newSet = []
            for obj in a_set:
                if isinstance(obj[1], music21.chord.Chord):
                    for note in obj[1]: # added
                        if not note.containerHierarchy:
                            note.containerHierarchy = obj[1].containerHierarchy
                        if not note.contextSites:
                            note.contextSites = obj[1].contextSites
                        if note.stemDirection == 'unspecified':
                            note.stemDirection = obj[1].stemDirection
                        newSet.append((obj[0], note))
                else:
                    newSet.append(obj)

            return newSet # modified

        def compareObj(aObj, bObj):
            # Compare Music 21 objects
            if aObj == bObj:
                return True
            if type(aObj) != type(bObj):
                if not isinstance(aObj, music21.key.Key) and not isinstance(aObj, music21.key.KeySignature): # added
                    return False
            if isinstance(aObj, music21.stream.Measure):
                return True
            if isinstance(aObj, music21.bar.Barline):
                return True
            if isinstance(aObj, music21.clef.Clef):
                if type(aObj) == type(bObj):
                    return True
            if isinstance(aObj, music21.key.Key) or isinstance(aObj, music21.key.KeySignature): # mod
                if aObj.sharps == bObj.sharps:
                    return True
            if isinstance(aObj, music21.meter.TimeSignature):
                if aObj.numerator == bObj.numerator and aObj.beatCount == bObj.beatCount:
                    return True
            if isinstance(aObj, music21.note.Note):
                if aObj.pitch == bObj.pitch and aObj.duration == bObj.duration and aObj.stemDirection == bObj.stemDirection:
                    return True
            if isinstance(aObj, music21.note.Rest):
                if aObj.duration == bObj.duration:
                    return True
            if isinstance(aObj, music21.chord.Chord):
                if aObj.duration == bObj.duration and set(aObj.pitches) == set(bObj.pitches) and aObj.stemDirection == bObj.stemDirection:
                    return True
            return False

        def findObj(aPair, a_set):
            # Find
            for bPair in a_set:
                if aPair[0] == bPair[0]:
                    if compareObj(aPair[1], bPair[1]):
                        return bPair
            return None

        def getBeams(note_obj): # added
            return '_'.join(['-'.join([b.type, b.direction]) if b.direction else b.type for b in note_obj.beams])

        def getTie(note_obj): # added
            return note_obj.tie.type if note_obj.tie is not None else ''

        def referClef(note_obj): # added
            context = note_obj.getContextByClass('Clef')
            return context.name if context is not None else ''

        def referTimeSig(note_obj): # added
            context = note_obj.getContextByClass('TimeSignature')
            return context.numerator / context.denominator \
                    if context is not None else ''

        def referKeySig(note_obj): # added
            keyObj = (note_obj.getContextByClass('Key') or note_obj.getContextByClass('KeySignature'))
            return keyObj.sharps if keyObj else 0

        def referVoice(note_obj): # added
            context = note_obj.getContextByClass('Voice')
            return context.id if context is not None else '1'

        errors = np.zeros((len(ScoreErrors.__members__)), int)

        a = a_set.copy()
        b = b_set.copy()

        # Remove matching pairs from both sets
        a_temp = []
        for pair in a:
            bPair = findObj(pair, b)
            if bPair:
                b.remove(bPair)
            else:
                a_temp.append(pair)
        a = a_temp

        # Find mismatched staff placement
        a_temp = []
        for obj in a:
            b_temp = [o[1] for o in b if o[0] != obj[0]]
            if obj[1] in b_temp:
                idx = b.index((1 - obj[0], obj[1]))
                del b[idx]
                errors[ScoreErrors.StaffAssignment] += 1
            else:
                a_temp.append(obj)
        a = a_temp

        a = splitChords(a)
        b = splitChords(b)

        # Find mismatches in notes
        a_temp = []
        for obj in a:
            if isinstance(obj[1], music21.note.Note):
                found = False
                for bObj in b:
                    if isinstance(bObj[1], music21.note.Note) and bObj[1].pitch == obj[1].pitch:
                        if bObj[0] != obj[0]:
                            errors[ScoreErrors.StaffAssignment] += 1
                        else: # added
                            if bObj[1].duration != obj[1].duration:
                                errors[ScoreErrors.NoteDuration] += 1
                            if bObj[1].stemDirection != obj[1].stemDirection:
                                errors[ScoreErrors.StemDirection] += 1

                            if getBeams(bObj[1]) != getBeams(obj[1]): # added
                                errors[ScoreErrors.Beams] += 1
                            if getTie(bObj[1]) != getTie(obj[1]): # added
                                errors[ScoreErrors.Tie] += 1
                            # if referClef(bObj[1]) != referClef(obj[1]): # added
                            #     errors[ScoreErrors.Clef] += 1
                            if referTimeSig(bObj[1]) != referTimeSig(obj[1]): # added
                                errors[ScoreErrors.TimeSignature] += 1
                            if referKeySig(bObj[1]) != referKeySig(obj[1]): # added
                                errors[ScoreErrors.KeySignature] += 1
                            # if referVoice(bObj[1]) != referVoice(obj[1]): # added
                            #     errors[ScoreErrors.Voice] += 1

                        b.remove(bObj)
                        found = True
                        break
                if not found:
                    a_temp.append(obj)
            else:
                a_temp.append(obj)
        a = a_temp

        # Find mismatched duration of rests
        a_temp = []
        for obj in a:
            if isinstance(obj[1], music21.note.Rest):
                for bObj in b:
                    if isinstance(bObj[1], music21.note.Rest) and bObj[1].duration != obj[1].duration:
                        b.remove(bObj)
                        errors[ScoreErrors.RestDuration] += 1
                        break
            a_temp.append(obj)
        a = a_temp

        # Find enharmonic equivalents and report spelling mistakes and duration mistakes
        a_temp = []
        for obj in a:
            if isinstance(obj[1], music21.note.Note):
                idx = findEnharmonicEquivalent(obj[1], b)
                if idx != -1:
                    if b[idx][0] != obj[0]:
                        errors[ScoreErrors.StaffAssignment] += 1
                    if b[idx][1].duration != obj[1].duration:
                        errors[ScoreErrors.NoteDuration] += 1
                    if b[idx][1].stemDirection != obj[1].stemDirection:
                        errors[ScoreErrors.StemDirection] += 1

                    if getBeams(b[idx][1]) != getBeams(obj[1]): # added
                        errors[ScoreErrors.Beams] += 1
                    if getTie(b[idx][1]) != getTie(obj[1]): # added
                        errors[ScoreErrors.Tie] += 1
                    # if referClef(b[idx][1]) != referClef(obj[1]): # added
                    #     errors[ScoreErrors.Clef] += 1
                    # if referTimeSig(b[idx][1]) != referTimeSig(obj[1]): # added
                    #     errors[ScoreErrors.TimeSignature] += 1
                    # if referKeySig(b[idx][1]) != referKeySig(obj[1]): # added
                    #     errors[ScoreErrors.KeySignature] += 1
                    # if referVoice(b[idx][1]) != referVoice(obj[1]): # added
                    #     errors[ScoreErrors.Voice] += 1

                    del b[idx]
                    errors[ScoreErrors.NoteSpelling] += 1
                else:
                    a_temp.append(obj)
            else:
                a_temp.append(obj)
        a = a_temp

        aErrors = countObjects(a)
        bErrors = countObjects(b)

        errors += bErrors
        errors[ScoreErrors.NoteInsertion] = aErrors[ScoreErrors.NoteDeletion]
        errors[ScoreErrors.RestInsertion] = aErrors[ScoreErrors.RestDeletion]
        return errors

    def getSet(aList, start, end):
        set = []
        for aTuple in aList:
            if aTuple[0] >= end:
                return set
            if aTuple[0] >= start:
                set += aTuple[1]
        return set

    path, _ = score_alignment(estScore, gtScore)

    aList = convertScoreToList(estScore)
    bList = convertScoreToList(gtScore)

    nSymbols = countSymbols(gtScore)

    errors = np.zeros((len(ScoreErrors.__members__)), float)

    aStart, aEnd = 0.0, 0.0
    bStart, bEnd = 0.0, 0.0
    for pair in path:
        if pair[0] != aEnd and pair[1] != bEnd:
            aEnd, bEnd = pair[0], pair[1]
            errors += compareSets(getSet(aList, aStart, aEnd), getSet(bList, bStart, bEnd))

            aStart, aEnd = aEnd, aEnd
            bStart, bEnd = bEnd, bEnd
        elif pair[0] == aEnd:
            bEnd = pair[1]
        else:
            aEnd = pair[0]

    errors += compareSets(getSet(aList, aStart, float('inf')), getSet(bList, bStart, float('inf')))

    results = {k: int(v) for k, v in zip(ScoreErrors.__members__.keys(), errors)}
    results.update(nSymbols)

    return results
