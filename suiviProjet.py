import pandas as pd
from flask import Flask, request, render_template_string

df = pd.read_excel("/NAS/tupac/romain/SuiviProjets.xlsx")
colonnes = df.columns.tolist()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        nouvelles_donnees = {}
        for col in colonnes:
            val = request.form.get(col)
            # Si "Autre", récupérer la valeur du champ texte associé
            if val == "Autre":
                val = request.form.get(f"{col}_autre")
            nouvelles_donnees[col] = val
        # Ajouter la nouvelle ligne au fichier Excel
        df_new = pd.DataFrame([nouvelles_donnees])
        df_new.to_excel("/NAS/tupac/romain/SuiviProjets.xlsx", mode="a", header=False, index=False)
        return "Données enregistrées !"

    # Générer le formulaire HTML
    formulaire_html = "<form method='post'>"
    for col in colonnes:
        if col in ["Début", "Fini"]:
            formulaire_html += f"<label>{col}</label>"
            formulaire_html += f"<input type='date' name='{col}'><br>"
        else:
            options = df[col].dropna().unique().tolist()
            formulaire_html += f"<label>{col}</label>"
            formulaire_html += f"<select name='{col}' onchange=\"if(this.value=='Autre'){{document.getElementById('{col}_autre').style.display='inline';}}else{{document.getElementById('{col}_autre').style.display='none';}}\">"
            for opt in options:
                formulaire_html += f"<option value='{opt}'>{opt}</option>"
            formulaire_html += f"<option value='Autre'>Autre</option></select>"
            formulaire_html += f"<input type='text' name='{col}_autre' id='{col}_autre' style='display:none;' placeholder='Autre...'><br>"
    formulaire_html += "<button type='submit'>Envoyer</button></form>"

    return render_template_string(formulaire_html)

if __name__ == "__main__":
    app.run(debug=True)