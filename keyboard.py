# KEYBOARD CONFIGURATIONS

typing_map ={
    'qwerty': {
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
    },
    'dvorak': {
      'left':
        {
          'pinkie' : ('a', ';'),
          'ring' : (',','o','q'),
          'middle' : ('.','e','j'),
          'index' : ('p','u','k','y','i','x')
        },
      'right':
        {
          'index' : ('f','d','b','g','h','m'),
          'middle' : ('c','t','w'),
          'ring' : ('r','n','v'),
          'pinkie' : ('l','s','z')
        }
    }
}



typing_row = {'qwerty': ['qwertyuiop','asdfghjkl','zxcvbnm','1234567890'],
              'dvorak': ['pyfgcrl','aoeuidhtns','qjkxbmwvz','1234567890']
             }




def left_hand(layout = 'qwerty'):
  return [c for cs in [typing_map[layout]['left'][finger] for finger in typing_map[layout]['left'].keys()] for c in cs]

def right_hand(layout = 'qwerty'):
  return [c for cs in [typing_map[layout]['right'][finger] for finger in typing_map[layout]['right'].keys()] for c in cs]
