# KEYBOARD CONFIGURATIONS

typing_map ={
    'querty': {
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
}



typing_row = {'qwerty': ['qwertyuiop','asdfghjkl','zxcvbnm','1234567890'],
              'dvorak': ['','','','']
             }




def left_hand(layout = 'qwerty'):
  return [c for cs in [typing_map[layout]['left'][finger] for finger in typing_map[layout]['left'].keys()] for c in cs]

def right_hand(layout = 'qwerty'):
  return [c for cs in [typing_map[layout]['right'][finger] for finger in typing_map[layout]['right'].keys()] for c in cs]
