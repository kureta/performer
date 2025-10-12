# Converts a MIDI file from LilyPond to use MIDI Tuning Standard messages
# instead of pitch bend.  Works with TiMidity++.

import os.path, sys, re, getopt

# Note unaffected by scale stretch:
centerPitch = 0x40

noteOnPattern = b'([\x90-\x9f][\x00-\x7f][\x01-\x7f])'
pitchBendPattern = b'([\xe0-\xef][\x00-\x7f][\x00-\x7f]\x00)'
tunedNotePattern = pitchBendPattern + noteOnPattern

# After pitch bends translated to MTS:
untunedNotePattern =  b'([^\xf7][\x00-\x7f])' + noteOnPattern

# Note off (captured) followed immediately by centered pitch bend
noteOffPattern = b'([\x90-\x9f][\x00-\x7f]\x00)\x00[\xe0-\xef]\x00\x40'

def applyMTS(track, stretch=1.0):
    channels = set()

    for bend, noteon in set(re.findall(tunedNotePattern, track)):
        channel, key = noteon[0]&0xf, noteon[1]
        channels.add(channel)
        assert (bend[0]&0xf) == channel
        coarse, fine = mtsTuning(key, bend, stretch)
        mts = mtsMessage(channel, key, coarse, fine)
        new = mts + b'\x00' + noteon
        track = track.replace(bend + noteon, new)

    for prelude, noteon in set(re.findall(untunedNotePattern, track)):
        channel, key = noteon[0]&0xf, noteon[1]
        channels.add(channel)
        coarse, fine = mtsTuning(key, None, stretch)
        mts = mtsMessage(channel, key, coarse, fine)
        new = prelude + mts + b'\x00' + noteon
        track = track.replace(prelude + noteon, new)

    rpns = b''.join([setTuningTable(channel) for channel in channels])
    return rpns + re.sub(noteOffPattern, b'\\1', track)

def mtsTuning(key, bend, stretch=1.0):
    if bend is None:
        pitch = float(key)
    else:
        bendSize = (bend[2]<<7) + bend[1] # Yes, bends are little endian
        # 14 bits total, 12 of fine tuning, 2 of coarse (+/- 2 semitones)
        pitch = float(key-2) + float(bendSize)/(1<<12)
    pitch = centerPitch + (pitch-centerPitch) * stretch
    coarse = int(pitch)
    # MTS fine tuning is 14 bits total
    fine = int((pitch-coarse) * (1<<14))
    return coarse, fine

def mtsMessage(channel, key, coarse, fine):
    program = channel
    device = 0x7f
    length, changes = 11, 1
    mts = bytes([0xf0, length, 0x7f, device, 0x08, 0x02,
        program, changes, key, coarse, fine>>7, fine&0x7f, 0xf7])
    assert len(mts) == length + 2
    return mts

def setTuningTable(channel):
    control = 0xB0 + channel
    MSB = [0, control, 101, 0]
    LSB = [0, control, 100, 3]
    return bytes(MSB + LSB + [0, control, 6, channel])

def convertFile(old, new, stretch=1.0):
    writeChunk(getChunk(old, b'MThd'), b'MThd', new)
    for track in getTracks(old):
        writeChunk(applyMTS(track, stretch), b'MTrk', new)

def getTracks(stream):
    while True:
        yield getChunk(stream, b'MTrk')
    assert stream.read() == b''

def getChunk(stream, tag):
    if stream.read(4) != tag:
        raise StopIteration('End of file')
    return stream.read(uint(stream.read(4)))

def writeChunk(midi, tag, output):
    output.write(tag)
    output.write(bigEndian32(len(midi)))
    output.write(midi)

def bigEndian32(i):
    """Integer to byte stream"""
    return bytes([x&0xFF for x in (i>>24, i>>16, i>>8, i)])

def uint(s):
    """Big endian byte stream to integer"""
    result = 0
    for byte in s:
        result = (result<<8) + byte
    return result

def stretchFromString(option):
    value, units = re.findall(r"^([\d.+-]+)(\S+)$", option)[0]
    value = float(value)
    if units in ("c", "cent", "cents", "cpo"):
        return 1.0 + value/1200
    if units == '%':
        return 1.0 + value/100
    if units in ("o", "oct", "octave", "octaves"):
        return value
    if not units:
        raise ValueError("Units for scale stretch are required.")
    raise ValueError("Unrecognized units for scale stretch: %r" % units)

if __name__=='__main__':
    opts, args = getopt.getopt(sys.argv[1:], "s:", ["stretch="])
    filename = args[0]
    if len(args) > 1:
        outname = args[1]
    else:
        root, ext = os.path.splitext(filename)
        outname = root + '-mts' + ext
    stretch = 1.0
    for key, value in opts:
        if key in ('-s', '--stretch'):
            stretch = stretchFromString(value)
        else:
            print(key)
    with open(filename, 'rb') as old:
        with open(outname, 'wb') as new:
            convertFile(old, new, stretch)
