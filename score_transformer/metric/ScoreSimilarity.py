from enum import IntEnum
import music21
import numpy as np
from numba import jit


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
    StaffAssignment = 10
    Voice = 11 # added



def score_alignment(score_a, score_b):
    """Compare two musical scores.

    Parameters:

        aScore/bScore: music21.stream.Score objects

    Return value:

        (path, d):
            path is a list of tuples containing pairs of matching offsets
            d is the alignment matrix
    """
    def score_to_pitch_offset_list(s):
        """Convert a piano score into a list of tuples containing pitches

        Parameter:
            s a music21.Stream containing two music21.stream.PartStaff

        Return value:
            np.ndarray, shape=(N,)
                offsets
            np.ndarray, shape=(N, 128) a binary array to indicate the presence of a pitch onset at that location
        """
        my_dict = {}
        for n in s.flatten().notes:
            if n.isChord:
                raise NotImplementedError("Chords are not supported")
            if n.offset not in my_dict:
                my_dict[n.offset] = np.zeros(128, dtype=np.uint16)
            my_dict[n.offset][n.pitch.midi] = 1
        offsets = list(my_dict.keys())
        pitches = list(my_dict.values())
        return np.array(offsets), np.array(pitches)

    @jit(nopython=True, cache=True)
    def costMatrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        d = np.zeros((len(s) + 1, len(t) + 1))
        d[1:, 0] = np.inf
        d[0, 1:] = np.inf
        for j in range(len(t)):
            for i in range(len(s)):
                d[i + 1, j + 1] = min(d[i, j + 1], d[i + 1, j], d[i, j]) + np.bitwise_xor(s[i], t[j]).sum()
                # d[i + 1, j + 1] = np.bitwise_xor(s[i], t[j]).sum()
        return d

    offsets_a, pitches_a = score_to_pitch_offset_list(score_a)
    offsets_b, pitches_b = score_to_pitch_offset_list(score_b)
    if not pitches_a.shape[0] or not pitches_b.shape[0]:
        return [(0, 0)]
    d = costMatrix(pitches_a, pitches_b)

    d_img = d
    d_img[np.isinf(d_img)] = d_img[np.logical_not(np.isinf(d_img))].max()
    d_img = (d_img / d_img.max() * 255).astype(np.uint8)


    i, j = (d.shape[0] - 1, d.shape[1] - 1)
    path = []
    while i and j:
        d_img[i-1, j-1] = 255
        path.insert(0, (offsets_a[i-1], offsets_b[j-1]))

        idx = np.argmin([d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]])
        if idx == 0:
            i = i - 1
        elif idx == 1:
            j = j - 1
        else:
            i, j = i - 1, j - 1

    # Mark path in image
    import time
    from PIL import Image
    # Image.fromarray(d_img, mode='L').save(f'd_{time.time()}.png')
    return path

def score_similarity(estScore: music21.stream.Score, gtScore: music21.stream.Score, partMapping={0: "right", 1: "left"}):
    """Compare two musical scores.

    Parameters:

    estScore/gtScore: music21.stream.Score objects of piano scores. The scores must contain two
        music21.stream.PartStaff substreams (top and bottom staves)

    estScore is the estimated transcription
    gtScore is the ground truth

    Return value:

    A NumPy array containing the differences between the two scores:
        barlines, clefs, key signatures, time signatures, note, note spelling,
        note duration, staff assignment.
    The differences for notes and barlines are normalized with the number of symbols
    in the ground truth.
    """

    if isinstance(estScore, str):
        estScore = music21.converter.parse(estScore).expandRepeats()
    if isinstance(gtScore, str):
        gtScore = music21.converter.parse(gtScore).expandRepeats()
    assert isinstance(estScore, music21.stream.Stream)
    assert isinstance(gtScore, music21.stream.Stream)


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
        CLASSES = (music21.bar.Barline, music21.note.Note, music21.chord.Chord)

        def convertStreamToList(s, hand):
            elements_with_offsets = {}
            for el in s.recurse():
                if isinstance(el, CLASSES):
                    offset = el.getOffsetInHierarchy(s)
                    elements_with_offsets.setdefault(offset, []).append((hand, el))
            return sorted(elements_with_offsets.items())

        def flattenStream(s: music21.stream.Stream):
            new_stream = music21.stream.Stream()
            for el in s.flatten():
                if isinstance(el, (music21.note.Note, music21.chord.Chord)):
                    new_stream.insert(el)
            for el in s.recurse():
                if isinstance(el, (music21.stream.Measure, music21.bar.Barline)):
                    new_stream.insert(el.getOffsetInHierarchy(s), music21.bar.Barline())
            return new_stream

        parts = aScore.getElementsByClass([music21.stream.PartStaff, music21.stream.Part])  # get staves
        topStaffList = []
        bottomStaffList = []
        validParts = True

        for i, part in enumerate(parts):
            if i not in partMapping:
                topStaffList += convertStreamToList(flattenStream(part), 0)
                validParts = False
                continue
            if partMapping[i] == "right":
                topStaffList += convertStreamToList(flattenStream(part), 0)
            else:
                bottomStaffList += convertStreamToList(flattenStream(part), 1)

        elTimes = []
        elList = []
        tIterator = iter(topStaffList)
        bIterator = iter(bottomStaffList)
        tEl = next(tIterator, None)
        bEl = next(bIterator, None)

        while tEl or bEl:
            if not tEl:
                elTimes.append(bEl[0])
                elList.append(bEl[1])
                bEl = next(bIterator, None)
            elif not bEl:
                elTimes.append(tEl[0])
                elList.append(tEl[1])
                tEl = next(tIterator, None)
            else:
                if tEl[0] < bEl[0]:
                    elTimes.append(tEl[0])
                    elList.append(tEl[1])
                    tEl = next(tIterator, None)
                elif tEl[0] > bEl[0]:
                    elTimes.append(bEl[0])
                    elList.append(bEl[1])
                    bEl = next(bIterator, None)
                else:
                    elTimes.append(tEl[0])
                    elList.append(tEl[1] + bEl[1])
                    tEl = next(tIterator, None)
                    bEl = next(bIterator, None)
        elTimes, elList = zip(*sorted(zip(elTimes, elList), key=lambda x: x[0]))

        return np.array(elTimes), elList, validParts

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

        def compareObj(aObj, bObj):
            # Compare Music 21 objects
            if aObj == bObj:
            # if recurse:
            #     print(aObj, bObj, compareObj(aObj, bObj, recurse=False))
                return True
            if isinstance(aObj, music21.key.Key) or isinstance(aObj, music21.key.KeySignature): # mod
                if aObj.sharps == bObj.sharps:
                    return True
            if type(aObj) != type(bObj):
                return False
            if isinstance(aObj, music21.stream.Measure):
                return True
            if isinstance(aObj, music21.bar.Barline):
                return True
            if isinstance(aObj, music21.clef.Clef):
                if type(aObj) == type(bObj):
                    return True
            if isinstance(aObj, music21.meter.TimeSignature):
                if aObj.numerator == bObj.numerator and aObj.beatCount == bObj.beatCount:
                    return True
            if isinstance(aObj, music21.note.Note):
                if aObj.pitch == bObj.pitch and abs(aObj.duration.quarterLength - bObj.duration.quarterLength) < 1e-3 and aObj.stemDirection == bObj.stemDirection:
                    return True
            return False

        def getBeams(note_obj: music21.note.Note): # added
            return '_'.join(['-'.join([b.type, b.direction]) if b.direction else b.type for b in note_obj.beams])

        def getTie(note_obj: music21.note.Note): # added
            return note_obj.tie.type if note_obj.tie is not None else ''

        def referClef(note_obj: music21.note.Note): # added
            context = note_obj.getContextByClass('Clef')
            return context.name if context is not None else ''

        def referTimeSig(note_obj: music21.note.Note): # added
            context = note_obj.getContextByClass(music21.meter.TimeSignature)
            return context.numerator / context.denominator \
                    if context is not None else ''

        def referKeySig(note_obj: music21.note.Note): # added
            keyObj = (note_obj.getContextByClass('Key') or note_obj.getContextByClass('KeySignature'))
            return keyObj.sharps if keyObj else 0

        def referVoice(note_obj: music21.note.Note): # added
            context = note_obj.getContextByClass('Voice')
            return context.id if context is not None else '1'

        errors = np.zeros((len(ScoreErrors.__members__)), int)

        a = a_set.copy()
        b = b_set.copy()

        # We filter a and b by removing matches from b, and building a new a that does not include the matches
        a_temp = []
        for pair in a:
            for bPair in b:
                if pair[0] == bPair[0] and compareObj(pair[1], bPair[1]):
                    b.remove(bPair)
                    break
            else:
                a_temp.append(pair)
        a = a_temp

        # Find mismatched staff placement
        # a_temp = []
        # for obj in a:
        #     b_temp = [o[1] for o in b if o[0] != obj[0]]
        #     if obj[1] in b_temp:
        #         idx = b.index((1 - obj[0], obj[1]))
        #         del b[idx]
        #         errors[ScoreErrors.StaffAssignment] += 1
        #     else:
        #         a_temp.append(obj)
        # a = a_temp

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
                            if abs(bObj[1].duration.quarterLength - obj[1].duration.quarterLength) > 1e-3:
                                errors[ScoreErrors.NoteDuration] += 1
                            if bObj[1].stemDirection != obj[1].stemDirection:
                                errors[ScoreErrors.StemDirection] += 1

                            if getBeams(bObj[1]) != getBeams(obj[1]): # added
                                errors[ScoreErrors.Beams] += 1
                            if getTie(bObj[1]) != getTie(obj[1]): # added
                                errors[ScoreErrors.Tie] += 1
                            # if referClef(bObj[1]) != referClef(obj[1]): # added
                            #     errors[ScoreErrors.Clef] += 1
                            # if referTimeSig(bObj[1]) != referTimeSig(obj[1]): # added
                            #     errors[ScoreErrors.TimeSignature] += 1
                            # if referKeySig(bObj[1]) != referKeySig(obj[1]): # added
                            #     errors[ScoreErrors.KeySignature] += 1
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

        # Find enharmonic equivalents and report spelling mistakes and duration mistakes
        a_temp = []
        for obj in a:
            if isinstance(obj[1], music21.note.Note):
                idx = findEnharmonicEquivalent(obj[1], b)
                if idx != -1:
                    if b[idx][0] != obj[0]:
                        errors[ScoreErrors.StaffAssignment] += 1
                    if abs(b[idx][1].duration.quarterLength - obj[1].duration.quarterLength) > 1e-3:
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
        # print("C", len(a), [n.offset for _, n in a], len(b), [n.offset for _, n in b])

        aErrors = countObjects(a)
        bErrors = countObjects(b)

        errors += bErrors
        errors[ScoreErrors.NoteInsertion] = aErrors[ScoreErrors.NoteDeletion]
        return errors

    def getSet(elements, times, start, end):
        endIdx = np.searchsorted(times, end, side='left')
        set = []
        for el in elements[start:endIdx]:
            set += el
        return set, endIdx

    path = score_alignment(estScore, gtScore)

    aTimes, aElements, aValidParts = convertScoreToList(estScore)
    bTimes, bElements, bValidParts = convertScoreToList(gtScore)
    if not (aValidParts and bValidParts):
        for i, aElement in enumerate(aElements):
            for j, el in enumerate(aElement):
                aElements[i][j] = (0, el[1])
        for i, bElement in enumerate(bElements):
            for j, el in enumerate(bElement):
                bElements[i][j] = (0, el[1])
    errors = np.zeros((len(ScoreErrors.__members__)), float)
    aIdx, aEnd = 0, 0.0
    bIdx, bEnd = 0, 0.0
    for pair in path:
        if pair[0] != aEnd and pair[1] != bEnd:
            aEnd = pair[0]
            bEnd = pair[1]
            aSet, aIdx = getSet(aElements, aTimes, aIdx, aEnd)
            bSet, bIdx = getSet(bElements, bTimes, bIdx, bEnd)
            errors += compareSets(aSet, bSet)
        elif pair[0] == aEnd:
            bEnd = pair[1]
        else:
            aEnd = pair[0]

    aSet, _ = getSet(aElements, aTimes, aIdx, float('inf'))
    bSet, _ = getSet(bElements, bTimes, bIdx, float('inf'))
    errors += compareSets(aSet, bSet)

    results = {k: int(v) for k, v in zip(ScoreErrors.__members__.keys(), errors)}
    results.update({"n_Note": len([n for n in gtScore.flatten().notes if not n.isRest])})
    if not (aValidParts and bValidParts):
        results['StaffAssignment'] = None
    return results
