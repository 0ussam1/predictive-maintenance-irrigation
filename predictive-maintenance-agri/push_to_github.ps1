# Script d'automatisation pour pousser le projet sur GitHub
# Auteur: Antigravity AI pour Oussama Akarrou

$repoName = "predictive-maintenance-irrigation"
$githubUser = "0ussam1"
$remoteUrl = "https://github.com/$githubUser/$repoName.git"

Write-Host "--- Organisation et Publication du Projet ---" -ForegroundColor Cyan

# 1. Vérification du dépôt local
if (!(Test-Path .git)) {
    Write-Host "[!] Initialisation du dépôt Git..."
    git init
}

# 2. Ajout du remote
Write-Host "[*] Configuration du lien vers GitHub ($remoteUrl)..."
git remote remove origin 2>$null
git remote add origin $remoteUrl

# 3. Branche main
git branch -M main

# 4. Push
Write-Host ""
Write-Host ">>> ATTENTION : Une fenêtre va s'ouvrir pour vous connecter à GitHub." -ForegroundColor Yellow
Write-Host ">>> Veuillez valider l'accès pour terminer la publication." -ForegroundColor Yellow
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] Votre projet est maintenant public sur GitHub !" -ForegroundColor Green
    Write-Host "Lien : https://github.com/$githubUser/$repoName" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[ERREUR] Le transfert a échoué. Vérifiez que vous avez créé le dépôt sur GitHub." -ForegroundColor Red
}

Write-Host "Appuyez sur une touche pour fermer..."
$null = [Console]::ReadKey()
