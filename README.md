# NOTE E VALUTAZIONI UTILI

## Pacchetti Installati
> I pacchetti installati manualmente con pip si trovano 
> sul file `requirements.txt`

Per l'utilizzo della libreria pdf2image è necesasrio installare
poppler. Per l'installazione su macchine Windows si segono
i seguenti passaggi:

    1. Scaricare i [binari dell'ultima release](https://github.com/oschwartz10612/poppler-windows/releases/tag/v23.11.0-0) 
    2. Estrarre i file dal pacchetto zip e salvarli in locale in `C:\Program Files`
    3. Aggiungere il path della cartella `bin` alla variabile di sistema `PATH`
    4. Per verifica corretta installazione: cmd > pdftoppm -h

Tesseract è utile per leggere le immagini all'interno di
documenti PDF. Per l'installazione su macchine Windows:

    1. Scaricare il file eseguibile per Tesseract OCR
    2. Installare il pacchetto software
    3. Aggiungere il path della cartella `Tesseract-OCR` alla variabile di sistema `PATH`
    4. Per verifica corretta installazione: cmd > tesseract

## Configurazione Account Google
Per abilitare i servizi Google Cloud è necessario aggiungere
la seguente variabile di ambiente.

> os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

In questa variabile viene salvato il percorso che fa riferimento
ad un file JSON scaricato dalla piattaforma Google Cloud Platform.
Questo file rappresenta una chiave che identifica un Service Account.
Il Service Account va registrato all'interno di un progetto su GCP
a cui devono essere assegnati i permessi necessari 
(ad esempio Vertex AI User per utilizzare i modelli di AI).

## Processing dei Documenti
### Estrazione del testo
Librerie utili per estrarre il testo sono **PyMuPDF** o **Unstructured**.
1. **PyMuPDF** - Vengono fornite delle classi per lavorare con i documenti
PDF in maniera piuttosto dettagliata. Ad esempio, le due classi
più utilizzate sono `Document` e `Page` che permettono di manipolare
documenti PDF su più livelli.
2. **Unstructured** - Vengono fornite funzionalità più orientate
al processing di file intesi per essere utilizzati in applicazioni Gen AI.
Infatti, al centro di questa libreria troviamo funzioni per fare
partitioning, cleaning, extracing, staging, chunking, embedding.
Ad esempio, vengono offerti diversi metodi di chunking ed in questo
progetto si è seguita la strategia "by_title" per ottenere chunk
rappresentativi dei paragrafi. La comodità di questa libreria è
proprio il fatto di avere metodi pensati per il processing di documenti
da inserire in vectore storage ad esempio. Altra differenza importante
è che in questo caso, possiamo lavorare su file di diversi formati.

### Estrazione delle tabelle
Anche in questo caso **Unstructured** è un'opzione. Infatti, si possono
configurare delle opzioni da passare alla funzione `partition_pdf` per
estrarre le tabelle. Il formato non è molto leggibile, infatti, riceviamo
una classe che rappresenta la tabella come una stringa.

La libreria **pdfplumber** permette di ottenere la tabella con
un formato più strutturato. Dato un documento PDF, possiamo
estrarre le tabelle in formato tabellare utilizzando le liste
come strutture dati. Questo permette di mantenere la struttura
logica della tabella in modo più leggibile.

### Estrazione delle immagini
Nell'eventualità che in un documento PDF siano presenti anche
immagini, la libreria **Unstructured** permette di estrarle in
modo automatico e salvarle in locale o su storage in cluod.

**PyMuPDF** offre un maggior grado di manipolazione delle immagini
in quanto mette a disposizione la classe Pixmap che permette di
lavorare sulle immagini anche a livello vettoriale. La libreria
offre molte informazioni sulle immagini estratte.
