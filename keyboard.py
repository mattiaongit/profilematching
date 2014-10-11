# KEYBOARD CONFIGURATIONS

typing_map = {
  'left':
    {
      'pinkie' : ('q', 'a', 'z'),
      'ring' : ('w','s','x'),
      'middle' : ('e','d','c'),
      'index' : ('r','f','v','t','g','b')
    },
  'right':
    {
      'index' : ('y','h','n','u','j','m'),
      'middle' : ('i','k',','),
      'ring' : ('o','l','.'),
      'pinkie' : ('p',';')
    }
  }


typing_row = ['qwertyuiop','asdfghjkl','zxcvbnm','1234567890']


def left_hand():
  return [c for cs in [typing_map['left'][finger] for finger in typing_map['left'].keys()] for c in cs]

def right_hand():
  return [c for cs in [typing_map['right'][finger] for finger in typing_map['right'].keys()] for c in cs]
