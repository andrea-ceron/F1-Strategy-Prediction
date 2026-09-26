import fastf1
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path


PARAMETRI = dict(passo_m=5, smoothing_m=65, soglia_ingresso=0.002,
                 soglia_uscita=0.001, ingresso_m=15, uscita_m=25,
                 cambio_verso_m=15)

def calcolo_interpolazione(posizioni, passo_interpolazione, passo_smoothing_savgol):
    xy = posizioni[['X [m]', 'Y [m]']].to_numpy(dtype='float')
    distance_between_each_sample = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(distance_between_each_sample)]
    n = int(np.ceil(s[-1]/passo_interpolazione))
    normalizzazione_spazio_per_interpolazione = np.linspace(0, s[-1], n, endpoint=False)
    passo = normalizzazione_spazio_per_interpolazione[1] - normalizzazione_spazio_per_interpolazione[0]
    numero_punti_finestra_savgol = int(round(passo_smoothing_savgol/ passo))
    numero_punti_finestra_savgol = max(5, numero_punti_finestra_savgol if numero_punti_finestra_savgol % 2 else numero_punti_finestra_savgol + 1)
    tempi_secondi = (posizioni['time'] - posizioni['time'].iloc[0]).dt.total_seconds().to_numpy()
    x, y = [np.interp(normalizzazione_spazio_per_interpolazione, s, xy[:, j]) for j in (0, 1)]
    tempi_interpolati = np.interp(normalizzazione_spazio_per_interpolazione, s, tempi_secondi)
    return x,y, numero_punti_finestra_savgol, passo, normalizzazione_spazio_per_interpolazione, tempi_interpolati

def calcolo_curvatura(x,y, numero_punti_finestra_savgol, passo, normalizzazione_spazio_per_interpolazione):
    def filtro(v, ordine=0):
        return savgol_filter(v, numero_punti_finestra_savgol, 3, deriv=ordine, delta=passo, mode='wrap')
    dx = filtro(x,1)
    dy = filtro(y,1)
    ddx = filtro(x,2)
    ddy = filtro(y,2)

    denominatore_curvatura = (dx**2 + dy**2)**1.5
    determinante = dx * ddy - dy * ddx

    curvatura = np.divide(determinante, denominatore_curvatura,
                          out=np.full(len(x), np.nan), where=denominatore_curvatura > 1e-9)
    tabella = pd.DataFrame({'X [m]': filtro(x), 'Y [m]': filtro(y), 'distanza_m': normalizzazione_spazio_per_interpolazione, 'curvatura': curvatura})
    tabella.attrs['passo_m'] = passo
    tabella.attrs['circuito_chiuso'] = True
    return tabella
    # plt.plot(tabella['X[m]'], tabella['Y[m]'])
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    # plt.title('Traiettoria del circuito')
    # plt.axis('equal')  # stessa scala sui due assi
    # plt.grid()
    # plt.show()
    # plt.plot(tabella['distanza'], tabella['curvatura'])
    # plt.xlabel('Distanza percorsa [m]')
    # plt.ylabel('Curvatura [1/m]')
    # plt.title('Curvatura lungo il circuito')
    # plt.grid()
    # plt.show()

def individua_curve(tabella,tempi_interpolati, soglia_ingresso=0.001, soglia_uscita=0.001,
                    ingresso_m=50, uscita_m=60, cambio_verso_m=20,
                    separa_stesso_verso=False):
    """Isteresi e persistenza spaziale; segno positivo = sinistra negli assi XY."""
    if min(ingresso_m, uscita_m, cambio_verso_m) <= 0:
        raise ValueError('Le distanze di conferma devono essere positive')
    curvature = tabella.curvatura.to_numpy()
    passo_campioni_m = tabella.attrs['passo_m']
    circuito_chiuso = tabella.attrs['circuito_chiuso']
    # Conferma su una distanza fra primo e ultimo campione >= valore richiesto.
    (
        campioni_richiesti_ingresso,
        campioni_richiesti_uscita,
        campioni_richiesti_cambio_verso,
    ) = [
        int(np.ceil(distanza_conferma_m / passo_campioni_m)) + 1
        for distanza_conferma_m in (ingresso_m, uscita_m, cambio_verso_m)
    ]
    curvature_da_analizzare = np.tile(curvature, 3) if circuito_chiuso else curvature
    gruppo_per_campione = np.zeros(len(curvature_da_analizzare), dtype=int)
    indice_inizio_curva = None
    verso_curva_attuale = 0
    # Conteggi di campioni consecutivi, azzerati quando la condizione si interrompe.
    consecutivi_ingresso_curva = 0
    consecutivi_bassa_curvatura = 0
    consecutivi_verso_opposto = 0
    verso_ingresso_candidato = 0
    numero_gruppo = 0
    for indice_campione, curvatura_campione in enumerate(curvature_da_analizzare):
        if not np.isfinite(curvatura_campione):
            if indice_inizio_curva is not None:
                numero_gruppo += 1
                gruppo_per_campione[indice_inizio_curva:indice_campione] = numero_gruppo
            indice_inizio_curva = None
            consecutivi_ingresso_curva = consecutivi_bassa_curvatura = consecutivi_verso_opposto = 0
            continue
        segno_curvatura = int(np.sign(curvatura_campione))
        if indice_inizio_curva is None:
            if abs(curvatura_campione) >= soglia_ingresso:
                consecutivi_ingresso_curva = (
                    consecutivi_ingresso_curva + 1
                    if segno_curvatura == verso_ingresso_candidato else 1
                )
                verso_ingresso_candidato = segno_curvatura
            else:
                consecutivi_ingresso_curva = 0
            if consecutivi_ingresso_curva >= campioni_richiesti_ingresso:
                indice_inizio_curva = indice_campione - consecutivi_ingresso_curva + 1
                verso_curva_attuale = segno_curvatura
                consecutivi_bassa_curvatura = consecutivi_verso_opposto = 0
            continue
        consecutivi_bassa_curvatura = (
            consecutivi_bassa_curvatura + 1
            if abs(curvatura_campione) <= soglia_uscita else 0
        )
        consecutivi_verso_opposto = (
            consecutivi_verso_opposto + 1
            if segno_curvatura == -verso_curva_attuale
            and abs(curvatura_campione) > soglia_uscita else 0
        )
        if consecutivi_bassa_curvatura >= campioni_richiesti_uscita:
            numero_gruppo += 1
            gruppo_per_campione[indice_inizio_curva:indice_campione - consecutivi_bassa_curvatura + 1] = numero_gruppo
            indice_inizio_curva = None
            consecutivi_ingresso_curva = consecutivi_bassa_curvatura = consecutivi_verso_opposto = 0
        elif consecutivi_verso_opposto >= campioni_richiesti_cambio_verso:
            indice_cambio_verso = indice_campione - consecutivi_verso_opposto + 1
            numero_gruppo += 1
            gruppo_per_campione[indice_inizio_curva:indice_cambio_verso] = numero_gruppo
            # Il verso opposto e gia confermato dentro un tratto in curva.
            indice_inizio_curva = indice_cambio_verso
            verso_curva_attuale = segno_curvatura
            consecutivi_bassa_curvatura = consecutivi_verso_opposto = 0
    if indice_inizio_curva is not None:
        gruppo_per_campione[indice_inizio_curva:] = numero_gruppo + 1
    if circuito_chiuso:
        gruppo_per_campione = gruppo_per_campione[len(curvature):2 * len(curvature)].copy()
        if (gruppo_per_campione[0] and gruppo_per_campione[-1] and gruppo_per_campione[0] != gruppo_per_campione[-1]
                and np.sign(curvature[0]) == np.sign(curvature[-1])):
            gruppo_per_campione[gruppo_per_campione == gruppo_per_campione[-1]] = gruppo_per_campione[0]
    print(gruppo_per_campione)
    # Secondo passaggio: due picchi nello stesso gruppo possono essere due curve.
    # Il minimo deve scendere sotto meta del picco minore; non basta una ondulazione.
    def ordina_indici(indici_gruppo):
        if circuito_chiuso and len(indici_gruppo) < len(curvature) and indici_gruppo[0] == 0 and indici_gruppo[-1] == len(curvature)-1:
            posizione_salto_traguardo = np.argmax(np.diff(indici_gruppo)) + 1
            return np.r_[indici_gruppo[posizione_salto_traguardo:], indici_gruppo[:posizione_salto_traguardo]]
        return indici_gruppo

    if separa_stesso_verso:
        ultimo_numero_gruppo = int(gruppo_per_campione.max())
        for gruppo_originale in pd.unique(gruppo_per_campione[gruppo_per_campione > 0]):
            indici_gruppo = ordina_indici(np.flatnonzero(gruppo_per_campione == gruppo_originale))
            modulo_curvatura_gruppo = np.abs(curvature[indici_gruppo])
            indici_picchi, _ = find_peaks(modulo_curvatura_gruppo, height=soglia_ingresso,
                                   prominence=soglia_uscita, distance=max(1, int(np.ceil(80 / passo_campioni_m))))
            posizioni_taglio = [0]
            for indice_primo_picco, indice_secondo_picco in zip(indici_picchi[:-1], indici_picchi[1:]):
                indice_minimo_tra_picchi = indice_primo_picco + int(np.argmin(modulo_curvatura_gruppo[indice_primo_picco:indice_secondo_picco+1]))
                if modulo_curvatura_gruppo[indice_minimo_tra_picchi] <= 0.3 * min(modulo_curvatura_gruppo[indice_primo_picco], modulo_curvatura_gruppo[indice_secondo_picco]):
                    posizioni_taglio.append(indice_minimo_tra_picchi)
            if len(posizioni_taglio) > 1:
                for indici_segmento in np.split(indici_gruppo, posizioni_taglio[1:]):
                    ultimo_numero_gruppo += 1
                    gruppo_per_campione[indici_segmento] = ultimo_numero_gruppo
    tabella_classificata = tabella.copy()
    tabella_classificata['gruppo'] = 0
    riepilogo_curve = []
    for numero_gruppo, gruppo_originale in enumerate(pd.unique(gruppo_per_campione[gruppo_per_campione > 0]), 1):
        indici_gruppo = ordina_indici(np.flatnonzero(gruppo_per_campione == gruppo_originale))
        tabella_classificata.loc[tabella_classificata.index[indici_gruppo], 'gruppo'] = numero_gruppo
        riepilogo_curve.append({'gruppo': numero_gruppo, 'inizio_m': float(tabella.distanza_m.iloc[indici_gruppo[0]]),
                       'fine_m': float(tabella.distanza_m.iloc[indici_gruppo[-1]]),
                       'lunghezza_m': float(len(indici_gruppo) * passo_campioni_m),
                       'attraversa_traguardo': bool(circuito_chiuso and (np.diff(indici_gruppo) < 0).any()),
                       'verso': 'sinistra' if np.nanmedian(curvature[indici_gruppo]) > 0 else 'destra'})
    tabella_classificata['tipo'] = np.where(tabella_classificata.gruppo > 0, 'curva', 'rettilineo')
    tabella_classificata['intervalli'] = tempi_interpolati
    tabella_classificata.loc[~np.isfinite(curvature), 'tipo'] = 'non_valido'
    return tabella_classificata, pd.DataFrame(riepilogo_curve, columns=['gruppo', 'inizio_m', 'fine_m',
                           'lunghezza_m', 'attraversa_traguardo', 'verso'])    

def salva_immagine(tabella, curve, percorso, titolo='Curve individuate'):
    """Salva una mappa senza aprire finestre; i numeri identificano i gruppi."""
    from matplotlib.figure import Figure
    figura = Figure(figsize=(10, 10))
    assi = figura.subplots()
    assi.plot(tabella['X [m]'], tabella['Y [m]'], color='lightgray', linewidth=2)
    for numero_gruppo in curve['gruppo']:
        maschera_gruppo = tabella.gruppo == numero_gruppo
        assi.plot(tabella['X [m]'].where(maschera_gruppo), tabella['Y [m]'].where(maschera_gruppo),
                color=f'C{(numero_gruppo - 1) % 10}', linewidth=3)
        punto_etichetta = tabella.loc[maschera_gruppo].iloc[len(tabella.loc[maschera_gruppo]) // 2]
        assi.annotate(str(numero_gruppo), (punto_etichetta['X [m]'], punto_etichetta['Y [m]']),
                    xytext=(6, 6), textcoords='offset points', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    assi.set_aspect('equal')
    assi.set_xlabel('X [m]')
    assi.set_ylabel('Y [m]')
    assi.set_title(f'{titolo}\n{len(curve)} segmenti candidati')
    figura.text(0.5, 0.02, 'Colori e numeri = gruppi, non curve ufficiali; grigio = tracciato di riferimento',
             ha='center', fontsize=9)
    figura.tight_layout(rect=(0, 0.05, 1, 1))
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(percorso, dpi=150)
    print(f'Immagine salvata: {percorso}')

def individua_rettilinei(tabella_classificata, lunghezza_giro_m=None, durata_giro_s=None):
    """Misura ogni rettilineo dal primo all'ultimo campione valido.

    La colonna 'intervalli' contiene tempi cumulati in secondi.
    I totali del giro servono per unire i tratti a cavallo del traguardo.
    """
    segmenti = []
    inizio = None
    for posizione, riga in enumerate(tabella_classificata.itertuples(index=False)):
        rettilineo = (
            riga.gruppo == 0 and riga.tipo == 'rettilineo'
            and np.isfinite(riga.distanza_m) and np.isfinite(riga.intervalli)
        )
        if rettilineo and inizio is None:
            inizio = posizione
        elif not rettilineo and inizio is not None:
            segmenti.append((inizio, posizione - 1))
            inizio = None
    if inizio is not None:
        segmenti.append((inizio, len(tabella_classificata) - 1))

    chiuso = tabella_classificata.attrs.get('circuito_chiuso', False)
    unisci_estremi = bool(
        chiuso and segmenti and segmenti[0][0] == 0
        and segmenti[-1][1] == len(tabella_classificata) - 1
    )
    if unisci_estremi:
        if (lunghezza_giro_m is None or durata_giro_s is None
                or not np.isfinite(lunghezza_giro_m) or lunghezza_giro_m <= 0
                or not np.isfinite(durata_giro_s) or durata_giro_s <= 0):
            raise ValueError('Per attraversare il traguardo servono lunghezza e durata del giro positive')
        if len(segmenti) > 1:
            segmento_traguardo = (segmenti[-1][0], segmenti[0][1])
            segmenti = [segmento_traguardo] + segmenti[1:-1]

    record = []
    for inizio, fine in segmenti:
        prima = tabella_classificata.iloc[inizio]
        ultima = tabella_classificata.iloc[fine]
        giro_intero = unisci_estremi and inizio == 0 and fine == len(tabella_classificata) - 1
        attraversa = inizio > fine or giro_intero
        lunghezza = float(ultima['distanza_m'] - prima['distanza_m'])
        durata = float(ultima['intervalli'] - prima['intervalli'])
        if giro_intero:
            lunghezza, durata = float(lunghezza_giro_m), float(durata_giro_s)
        elif attraversa:
            lunghezza += lunghezza_giro_m
            durata += durata_giro_s
        record.append({
            'rettilineo': len(record) + 1,
            'inizio_m': float(prima['distanza_m']),
            'fine_m': float(ultima['distanza_m']),
            'lunghezza_m': lunghezza,
            'durata_s': durata,
            'velocita_media_kmh': lunghezza / durata * 3.6 if durata > 0 else np.nan,
            'attraversa_traguardo': attraversa,
        })
    return pd.DataFrame(record, columns=[
        'rettilineo', 'inizio_m', 'fine_m', 'lunghezza_m',
        'durata_s', 'velocita_media_kmh', 'attraversa_traguardo'
    ])

def estrazione_feature_circuito(anno, circuito, tipo):
    session = fastf1.get_session(anno, circuito, tipo)
    session.load(telemetry=True, weather=False, messages=False)
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    posizioni = pd.DataFrame({'X [m]': pos.X / 10, 'Y [m]': pos.Y / 10, 'time': pos.Time})
    print(posizioni)
    x,y, numero_punti_finestra_savgol, passo, normalizzazione_spazio_per_interpolazione, tempi_interpolati = calcolo_interpolazione(posizioni, 5, 55)
    tabella = calcolo_curvatura(x,y, numero_punti_finestra_savgol, passo, normalizzazione_spazio_per_interpolazione)
    tabella_classificata, curve = individua_curve(tabella, tempi_interpolati)
    print(tabella_classificata.to_string())
    # Totali del tratto originale usato come giro, prima del ricampionamento.
    rettilinei = individua_rettilinei(
        tabella_classificata,
        lunghezza_giro_m=len(x) * passo,
        durata_giro_s=(posizioni['time'].iloc[-1] - posizioni['time'].iloc[0]).total_seconds()
    )
    print(rettilinei.to_string(index=False))

    salva_immagine(tabella_classificata, curve, Path(__file__).parent / 'output' / f'curve_individuate_{circuito}_{anno}_{tipo} .png')

def main():
    estrazione_feature_circuito(2025, 'Las Vegas Grand Prix', 'Q')
    


if __name__ == '__main__':
    main()
