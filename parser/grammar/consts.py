VAR_NAME='A'
SLOT_PREFIX='var'
TYPE_SIGN='$'
STRING_FIELD='STRING'
END_TOKEN = '</f>'
NT_TYPE_SIGN='#'
ROOT = 'answer'
IMPLICIT_HEAD = 'hidden'
FULL_STACK_LENGTH = 100000
NT = 'nonterminal'
PAD_PREDICATE = 'pad_predicate'
UI_BUTTON = '#UI_BUTTON#'


SPECIAL_TOKENS = [']', '[', '(', ')', ',', ':']


# supervised attention
PREDICATE_LEXICON = {
    'loc' : ['location', 'where', 'in', 'loc'],
    'argmax' : ['highest', 'largest', 'most', 'greatest', 'longest', 'biggest', "high", "maximum"],
    'argmin' : ['shortest', 'smallest', 'least', 'lowest', 'minimum'],
    '>' : ['greater', 'larger', '-er', 'than'],
    'count' : ['many','count'],
    '=': ['equal'],
    '\+' : ['not'],
    'req' : ['require'],
    'deg' : ['degree'],
    'exp' : ['experience'],
    'des' : ['desire'],
    #'deg' : ['degree'],
    'len' : ['length'],
    'next' : ['next', 'border', 'border', 'neighbor', 'surround'],
    #'density': ['densiti', 'averag', 'popul','density'],
    'density': ['average', 'population','density'],
    'sum': ['total', 'sum'],
    'size': ['size', 'big', 'biggest', 'largest'],
    'population': ['population', 'people', 'citizen']
    #'population': ['population', 'peopl', 'popul', 'citizen']
}

# preprocess data to turn stem into lemma
STEM_LEXICON = {
'desir': 'desire',
'posit': 'position',
'ha':'has',
'doe': 'does',
'citi': 'city',
'kilomet' : 'kilometer',
'averag' : 'average',
'squar' : 'square',
'popul': 'population',
'peopl': 'people',
'capit': 'capital',
'locat' : 'location',
'densiti': 'density',
'leav' : 'leave',
'arriv' : 'arrive',
'airlin' : 'airline',
'earli' : 'early',
'expens' : 'expensive',
'servic' : 'service',
'daili' : 'daily' ,
'avail': 'available',
'economi' : 'economic' ,
'schedul' : 'schedule',
'possibl' : 'possible',
'arrang' : 'arrange',
'airfar' : 'airfare',
'onli' : 'only',
'morn': 'morning',
'departur' : 'departure',
'airplan' :'airplane',
'abbrevi' : 'abbreviation',
'layov' : 'layover',
'stopov' : 'stopover',
'distanc' : 'distance',
'anywher' : 'anywhere',
'reserv' : 'reserve',
'oper':'operate',
'passeng' : 'passenger',
'describ' : 'describe',
'choic' : 'choice',
'anoth' : 'another',
'sorri' :'sorry',
'defin' : 'define',
'sometim' : 'sometimes',
'databas' : 'database',
'texa' : 'texas',
'approxim' : 'approximate',
'transcontinent' : 'transcontinental',
'itinerari' : 'itinerary',
'compani' : 'company',
'everywher' : 'everywhere',
'inexpens' : 'inexpensive',
'besid' : 'beside',
'charg' : 'charge',
'econom' : 'economy',
'qualifi' : 'qualify',
'advertis' : 'advertise',
'singl' : 'single',
'continu' : 'continue',
'somebodi' : 'somebody',
'mealtim' : 'mealtime',
'nighttim' : 'nighttime',
'thereaft' : 'thereafter',
'determin' : 'determine',
'requir' : 'require',
'knowledg' : 'knowledge',
'degre' : 'degree',
'experi' : 'experience',
'comput' : 'computer',
'involv' : 'involve',
'specialti' : 'specialty',
'everyth' : 'everything',
'applic' : 'application',
'administr' : 'administration',
'outsid' : 'outside',
'requirng' : 'requiring',
'someth' : 'something',
'titl' : 'title',
'anyth' : 'anything',
'salari': 'salary',
'mani': 'many',
'ani' : 'any',
'colleg': 'college',
'serv': 'serve',
'busi' : 'busy'
}

