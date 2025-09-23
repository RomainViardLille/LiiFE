from flask import Flask, request, render_template_string, redirect, url_for, session
import pandas as pd

app = Flask(__name__)
app.secret_key = "secret"  # Nécessaire pour utiliser session

df = pd.read_excel("/NAS/tupac/romain/NextCloud_UNIV/AmeliorationContinue_PTD_perso/SuiviProjets.xlsx")
colonnes = df.columns.tolist()

@app.route("/", methods=["GET", "POST"])
def form():
    # Récupère l'index courant (par défaut 0)
    idx = int(request.args.get("idx", session.get("idx", 0)))
    session["idx"] = idx

    # Pré-remplissage
    valeurs = df.iloc[idx].to_dict() if len(df) > 0 else {col: "" for col in colonnes}

    if request.method == "POST":
        nouvelles_donnees = {}
        for col in colonnes:
            val = request.form.get(col)
            if val == "Autre":
                val = request.form.get(f"{col}_autre")
            nouvelles_donnees[col] = val
        df.loc[idx] = nouvelles_donnees  # Met à jour la ligne
        #df.to_excel("/NAS/tupac/romain/NextCloud_UNIV/AmeliorationContinue_PTD_perso/SuiviProjets.xlsx", index=False)
        return redirect(url_for("form", idx=idx))

    # Navigation
    bouton_nav = ""
    if idx > 0:
        bouton_nav += f'<a href="?idx={idx-1}"><button>&lt;</button></a>'
    if idx < len(df)-1:
        bouton_nav += f'<a href="?idx={idx+1}"><button>&gt;</button></a>'

    # Entête HTML
    with open("/NAS/tupac/romain/NextCloud_UNIV/AmeliorationContinue_PTD_perso/SA_LIIFE_FOR_PTD_001_entete_FicheProjet.html", "r") as f:
        entete_html = f.read().rstrip()
    formulaire_html = entete_html.replace("Fiche projet n°", f"Fiche projet n°{idx+1:05d}") + bouton_nav
    formulaire_html += "<form method='post'>"

    formulaire_html += f"<h2>{df.loc[idx]['Nom du projet']}</h2>"
    # Colonnes "." sur une seule ligne
    dot_cols = [col for col in colonnes if col.startswith(".")]
    if dot_cols:
        formulaire_html += "<div style='margin-bottom:20px; display:flex; gap:20px;'>"
        for col in dot_cols:
            val = valeurs.get(col, "")
            checked = ""
            if pd.notna(val) and str(val) != "0":
                checked = "checked"
            formulaire_html += f"<label style='display:flex;align-items:center;'><input type='checkbox' name='{col}' value='1' {checked}> {col}</label>"
        formulaire_html += "</div>"
    
    # Colonnes "Type" à garder en liste déroulante
    type_cols = [col for col in colonnes if col.startswith("Type")]
    if type_cols:
        formulaire_html += "<div style='margin-bottom:20px; display:flex; gap:20px;'>"
        for col in type_cols:
            print(col)            
            options = df[col].dropna().unique().tolist()
            formulaire_html += f"<label>{col} :</label>"
            formulaire_html += f"<select name='{col}' onchange=\"if(this.value=='Autre'){{document.getElementById('{col}_autre').style.display='inline';}}else{{document.getElementById('{col}_autre').style.display='none';}}\">"
            for opt in options:
                selected = "selected" if val == opt else ""
                formulaire_html += f"<option value='{opt}' {selected}>{opt}</option>"
            formulaire_html += f"<option value='Autre'>Autre</option></select>"
            formulaire_html += f"<input type='text' name='{col}_autre' id='{col}_autre' style='display:none;' placeholder='Autre...'></div>"
        formulaire_html += "</div>"
    
    # Colonnes "_" sur une seule ligne pour le personnel
    underscore_cols = [col for col in colonnes if col.startswith("_")]

    # Autres colonnes
    for col in set(colonnes) - set(dot_cols) - set(underscore_cols) -set(type_cols):
        val = valeurs.get(col, "")
        if col in ["Début", "Fini"]:
            formulaire_html += f"<div style='margin-bottom:20px;'><label>{col} :</label>"
            formulaire_html += f"<input type='date' name='{col}' value='{val if pd.notna(val) else ''}'></div>"
        else:
            options = df[col].dropna().unique().tolist()
            formulaire_html += f"<div style='margin-bottom:20px;'><label>{col} :</label>"
            formulaire_html += f"<select name='{col}' onchange=\"if(this.value=='Autre'){{document.getElementById('{col}_autre').style.display='inline';}}else{{document.getElementById('{col}_autre').style.display='none';}}\">"
            for opt in options:
                selected = "selected" if val == opt else ""
                formulaire_html += f"<option value='{opt}' {selected}>{opt}</option>"
            formulaire_html += f"<option value='Autre'>Autre</option></select>"
            formulaire_html += f"<input type='text' name='{col}_autre' id='{col}_autre' style='display:none;' placeholder='Autre...'></div>"

    # Colonnes "_" sur une seule ligne pour le personnel
    if underscore_cols:
        formulaire_html += "<div style='margin-bottom:20px; display:flex; gap:20px;'>"
        for col in underscore_cols:
            val = valeurs.get(col, "")
            checked = ""
            if pd.notna(val) and str(val) != "0":
                checked = "checked"
            formulaire_html += f"<label style='display:flex;align-items:center;'><input type='checkbox' name='{col}' value='1' {checked}> {col}</label>"
        formulaire_html += "</div>"

    formulaire_html += "<button type='submit'>Envoyer</button></form>"

    with open("/NAS/tupac/romain/NextCloud_UNIV/AmeliorationContinue_PTD_perso/SA_LIIFE_FOR_PTD_001_pied_FicheProjet.html", "r") as f:
        pied_html = f.read()
    # print(formulaire_html)
    return render_template_string(formulaire_html + pied_html)

if __name__ == "__main__":
    app.run(debug=True)