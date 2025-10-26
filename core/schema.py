BUNDESLAENDER = [
    "Baden-Württemberg","Bayern","Berlin","Brandenburg","Bremen","Hamburg","Hessen",
    "Mecklenburg-Vorpommern","Niedersachsen","Nordrhein-Westfalen","Rheinland-Pfalz",
    "Saarland","Sachsen","Sachsen-Anhalt","Schleswig-Holstein","Thüringen"
]

KATEGORIEN = ["Innovation","Investition","Finanzierung"]

# Order is important. 'gruendungsjahr' will be asked ONLY if kategorie == "Innovation".
REQUIRED_FIELDS = [
    ("kategorie", "Thema / Kategorie (Innovation, Investition, Finanzierung)"),
    ("branche", "In welcher Branche sind Sie aktiv?"),
    ("bundesland", "In welchem Bundesland ist das Unternehmen ansässig?"),
    ("gruendungsjahr", "In welchem Jahr wurde das Unternehmen gegründet?"),
    ("projektkosten_eur", "Wie hoch sind die Projektkosten in Euro?"),
    ("eigenanteil_eur", "Wie viel Eigenanteil können Sie aufbringen?"),
]
