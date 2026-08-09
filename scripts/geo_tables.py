"""
Shared geocoding reference tables.

Single source of truth for the location-cleaning tables. Every script that
geocodes imports from here, so a correction made once applies everywhere.
(Duplicated copies of these tables previously drifted apart and produced
subtly different samples between steps.)

  US_STATES     - US postal codes, routed to ", USA"
  CA_PROVS      - Canadian provinces. NOTE: 'NT' is deliberately absent;
                  in this dataset NT means the Netherlands, not Northwest
                  Territories (Amsterdam, Apeldoorn, Rosmalen all appear
                  with NT).
  COUNTRY_MAP   - two-letter codes used in the source data -> country name
  TYPO_FIXES    - (city, ST) -> (city, ST) corrections, built from repeated
                  audits of the geocode failure log
  MANUAL_COORDS - hand-verified coordinates for real places Nominatim lacks
"""

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC'
}

CA_PROVS = {'ON', 'QC', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB', 'NL', 'PE', 'YT', 'NU'}

COUNTRY_MAP = {
    'PR': 'Puerto Rico', 'DR': 'Dominican Republic', 'VE': 'Venezuela',
    'VZ': 'Venezuela', 'MX': 'Mexico', 'CU': 'Cuba', 'CB': 'Cuba',
    'JA': 'Japan', 'JP': 'Japan', 'KO': 'South Korea', 'AU': 'Australia',
    'AS': 'American Samoa', 'VI': 'US Virgin Islands', 'GU': 'Guam',
    'JM': 'Jamaica', 'PN': 'Panama', 'NI': 'Nicaragua', 'BZ': 'Belize',
    'IT': 'Italy', 'SG': 'Singapore', 'PH': 'Philippines', 'NZ': 'New Zealand',
    'VG': 'British Virgin Islands', 'EN': 'England', 'HO': 'Honduras',
    'SP': 'Spain', 'GE': 'Germany',
    # --- added after auditing the failure log ---
    'NT': 'Netherlands',      # NOT Northwest Territories in this dataset
    'SR': 'Saudi Arabia',     # Dammam, Dhahran, Jeddah
    'UE': 'Ukraine',          # Kiev/Kyiv
    'SX': 'Eswatini',         # Mbabane
    'BR': 'Barbados',         # verify: could be Brazil in other datasets
    'CO': 'Colombia',
    'AR': 'Argentina',
    'TW': 'Taiwan',
    'KR': 'South Korea',
    'CN': 'China',
}

TYPO_FIXES = {
    # ------------------------------------------------------------------
    # Built from the geocode failure audit. Format: (city, ST) -> (city, ST)
    #
    # GROUP A — spelling corrections, state unchanged. High confidence.
    # ------------------------------------------------------------------
    ('Ahwattukee', 'AZ'): ('Ahwatukee', 'AZ'),
    ('Angelton', 'TX'): ('Angleton', 'TX'),
    ('Camana Island', 'WA'): ('Camano Island', 'WA'),
    ('Connorsville', 'IN'): ('Connersville', 'IN'),
    ('Couer D Alene', 'ID'): ("Coeur d'Alene", 'ID'),
    ('De Bary', 'FL'): ('DeBary', 'FL'),
    ('Donnelison', 'IL'): ('Donnellson', 'IL'),
    ('Duckhill', 'MS'): ('Duck Hill', 'MS'),
    ('Edmos', 'WA'): ('Edmonds', 'WA'),
    ('Ellsville', 'MO'): ('Ellisville', 'MO'),
    ('Florisville', 'TX'): ('Floresville', 'TX'),
    ("Floyd's Knob", 'IN'): ('Floyds Knobs', 'IN'),
    ('Galliopolis', 'OH'): ('Gallipolis', 'OH'),
    ('Graffton', 'VA'): ('Grafton', 'VA'),
    ('Hamiton', 'OH'): ('Hamilton', 'OH'),
    ('Hattiestburg', 'MS'): ('Hattiesburg', 'MS'),
    ('Highalnd', 'MI'): ('Highland', 'MI'),
    ('Highstown', 'NJ'): ('Hightstown', 'NJ'),
    ('Keeau', 'HI'): ('Keaau', 'HI'),
    ('Kerryville', 'TX'): ('Kerrville', 'TX'),
    ('Lapear', 'MI'): ('Lapeer', 'MI'),
    ('Longuieuil', 'QC'): ('Longueuil', 'QC'),
    ('Mereaux', 'LA'): ('Meraux', 'LA'),
    ('Mount Belvieu', 'TX'): ('Mont Belvieu', 'TX'),
    ('Mufreesboro', 'TN'): ('Murfreesboro', 'TN'),
    ('Nicholsville', 'KY'): ('Nicholasville', 'KY'),
    ('Oscutt', 'CA'): ('Orcutt', 'CA'),
    ('Rollings Hills Estates', 'CA'): ('Rolling Hills Estates', 'CA'),
    ('Roncevert', 'WV'): ('Ronceverte', 'WV'),
    ('Shamakin', 'PA'): ('Shamokin', 'PA'),
    ('Twins Lakes', 'WI'): ('Twin Lakes', 'WI'),
    ('Verdale', 'WA'): ('Veradale', 'WA'),
    ('Vinyard Haven', 'MA'): ('Vineyard Haven', 'MA'),
    ('Wailuki', 'HI'): ('Wailuku', 'HI'),
    ('West Windor', 'NJ'): ('West Windsor', 'NJ'),
    ('Willingsboro', 'NJ'): ('Willingboro', 'NJ'),
    ('Woodstrock', 'NB'): ('Woodstock', 'NB'),
    ('Tightaqueeze', 'VA'): ('Tightsqueeze', 'VA'),
    ('Layhoma', 'OK'): ('Lahoma', 'OK'),

    # ------------------------------------------------------------------
    # GROUP B — international spelling / country handling.
    # ------------------------------------------------------------------
    ('Enseneda', 'MX'): ('Ensenada', 'MX'),
    ('La Pieded', 'MX'): ('La Piedad', 'MX'),
    ('Zacatenas', 'MX'): ('Zacatecas', 'MX'),
    ('San Sebastien', 'PR'): ('San Sebastian', 'PR'),
    ('Toabaja', 'PR'): ('Toa Baja', 'PR'),
    ('Guayamas', 'PR'): ('Guayama', 'PR'),
    ('Los Alcanzzas', 'DR'): ('Los Alcarrizos', 'DR'),
    ('Badkissagen', 'GE'): ('Bad Kissingen', 'GE'),
    ('Mbabne', 'SX'): ('Mbabane', 'SX'),
    ('Jaddah', 'SR'): ('Jeddah', 'SR'),
    ('Kiev', 'UE'): ('Kyiv', 'UE'),
    ('Frederiksted', 'VG'): ('Frederiksted', 'VI'),   # St. Croix is US VI, not British
    ('Panama Canal Zone', 'PN'): ('Balboa', 'PN'),
    ('Buradados', 'BR'): ('Bridgetown', 'BR'),        # verify

    # ------------------------------------------------------------------
    # GROUP C — the STATE looks wrong, not the city name. These are the
    # judgment calls: the city exists, but not in the state the source
    # data gives. Verify against the player's record before trusting.
    # Comment any line out to leave it unresolved instead.
    # ------------------------------------------------------------------
    ('Blountville', 'TX'): ('Blountville', 'TN'),
    ('Crawfordsville', 'PA'): ('Crawfordsville', 'IN'),
    ('Fort Oglethorpe', 'FL'): ('Fort Oglethorpe', 'GA'),
    ('Siouz City', 'AL'): ('Sioux City', 'IA'),
    ('South Elgin', 'OH'): ('South Elgin', 'IL'),
    ('Summersville', 'OR'): ('Summersville', 'WV'),
    ('Glace Bay', 'NB'): ('Glace Bay', 'NS'),
    ('Grandville', 'MO'): ('Grandville', 'MI'),
    ('Ranchuelo', 'DR'): ('Ranchuelo', 'CU'),         # Ranchuelo is in Cuba

    # ------------------------------------------------------------------
    # GROUP D — final audit round. Clear misspellings, high confidence.
    # ------------------------------------------------------------------
    ('Brookiet', 'GA'): ('Brooklet', 'GA'),
    ('Demerest', 'NJ'): ('Demarest', 'NJ'),
    ('Demson', 'TX'): ('Denison', 'TX'),
    ('Gadisen', 'AL'): ('Gadsden', 'AL'),
    ('Kirskville', 'MO'): ('Kirksville', 'MO'),
    ('Manuta', 'OH'): ('Mantua', 'OH'),
    ('McRain', 'MI'): ('McBain', 'MI'),
    ('Pocolan', 'OK'): ('Pocola', 'OK'),
    ('Warrenville Heights', 'OH'): ('Warrensville Heights', 'OH'),
    # Painesville OH is a real Lake County city; Paintsville is in KY.
    # State is OH in the source, so Painesville is the likelier intent.
    ('Paintsville', 'OH'): ('Painesville', 'OH'),
    # Stollings WV (Logan County). "Stallings" is a NC town, not WV.
    ('Stallings', 'WV'): ('Stollings', 'WV'),
    # Christiansted is on St. Croix = US Virgin Islands, not British.
    ('Christiansted', 'VG'): ('Christiansted', 'VI'),
    # Xavier Cedeno (2004 Rockies): "Desal" is a fragment of his school name
    # (Asuncion Rodriguez De Sala School). His actual town is Guayanilla.
    ('Desal', 'PR'): ('Guayanilla', 'PR'),
    # Matt Harter (2000 Orioles): Penns Valley Area HS is in Spring MILLS, PA.
    ('Spring Hills', 'PA'): ('Spring Mills', 'PA'),

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STILL UNRESOLVED (intentional):
    #   ('--', '--')   placeholder in the source data, no location
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # GROUP E - full-register (1965-2025) audit round. High confidence.
    # ------------------------------------------------------------------
    ('Connorsville', 'IN'): ('Connersville', 'IN'),
    ('Couer D Alene', 'ID'): ("Coeur d'Alene", 'ID'),
    ('De Bary', 'FL'): ('DeBary', 'FL'),
    ('Duckhill', 'MS'): ('Duck Hill', 'MS'),
    ('Edmos', 'WA'): ('Edmonds', 'WA'),
    ('Ellsville', 'MO'): ('Ellisville', 'MO'),
    ("Floyd's Knob", 'IN'): ('Floyds Knobs', 'IN'),
    ('Galliopolis', 'OH'): ('Gallipolis', 'OH'),
    ('Graffton', 'VA'): ('Grafton', 'VA'),
    ('Hamiton', 'OH'): ('Hamilton', 'OH'),
    ('Hattiestburg', 'MS'): ('Hattiesburg', 'MS'),
    ('Highalnd', 'MI'): ('Highland', 'MI'),
    ('Highstown', 'NJ'): ('Hightstown', 'NJ'),
    ('Keeau', 'HI'): ('Keaau', 'HI'),
    ('Lapear', 'MI'): ('Lapeer', 'MI'),
    ('Mereaux', 'LA'): ('Meraux', 'LA'),
    ('Mount Belvieu', 'TX'): ('Mont Belvieu', 'TX'),
    ('Mufreesboro', 'TN'): ('Murfreesboro', 'TN'),
    ('Nicholsville', 'KY'): ('Nicholasville', 'KY'),
    ('Rollings Hills Estates', 'CA'): ('Rolling Hills Estates', 'CA'),
    ('Roncevert', 'WV'): ('Ronceverte', 'WV'),
    ('Shamakin', 'PA'): ('Shamokin', 'PA'),
    ('Twins Lakes', 'WI'): ('Twin Lakes', 'WI'),
    ('Wailuki', 'HI'): ('Wailuku', 'HI'),
    ('Willingsboro', 'NJ'): ('Willingboro', 'NJ'),
    # state corrections (city right, state wrong):
    ('Siouz City', 'AL'): ('Sioux City', 'IA'),
    # foreign / territory misspellings:
    ('Enseneda', 'MX'): ('Ensenada', 'MX'),
    ('La Pieded', 'MX'): ('La Piedad', 'MX'),
    ('Zacatenas', 'MX'): ('Zacatecas', 'MX'),
    ('Longuieuil', 'QC'): ('Longueuil', 'QC'),
    ('Woodstrock', 'NB'): ('Woodstock', 'NB'),
    ('Guayamas', 'PR'): ('Guayama', 'PR'),
    ('San Sebastien', 'PR'): ('San Sebastian', 'PR'),
    ('Jaddah', 'SR'): ('Jeddah', 'SR'),

}

MANUAL_COORDS = {
    # Unincorporated community in Pittsylvania County, VA.
    # Source: Wikipedia/GeoHack 36 46 52.2 N, 79 23 49.4 W
    ('Tightsqueeze', 'VA'): (36.781167, -79.397056),
}


def build_query(city, state):
    """Turn a (city, state) pair into a Nominatim query string."""
    if not city or not state:
        return None
    city, state = str(city).strip(), str(state).strip()
    if state in US_STATES:
        return f"{city}, {state}, USA"
    if state in CA_PROVS:
        return f"{city}, {state}, Canada"
    if state in COUNTRY_MAP:
        return f"{city}, {COUNTRY_MAP[state]}"
    return f"{city}, {state}"


def apply_typo_fix(city, state):
    """Return the corrected (city, state), or the original if no fix applies."""
    if city is None or state is None:
        return city, state
    key = (str(city).strip(), str(state).strip())
    return TYPO_FIXES.get(key, key)
