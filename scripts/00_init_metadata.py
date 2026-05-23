"""
Generates a ready-to-fill metadata.csv at data/personal/metadata.csv.

Each row corresponds to one chunk of the recording text. Add a row per speaker
per chunk:  copy a row, change filename and speaker, set difficulty.
"""

import pandas as pd
from pathlib import Path

CHUNKS = {
    "001": "pues mira te voy a contar cómo es mi día a día porque la verdad es que desde que cambié algunas cosas en mi rutina me siento bastante mejor antes me costaba un montón levantarme por las mañanas o sea sonaba el despertador y yo seguía tumbado sin hacer nada",
    "002": "ahora intento levantarme más o menos a las siete y media que sé que para mucha gente eso es tardísimo pero para mí ya es un avance lo primero que hago antes incluso de mirar el móvil es beberme un vaso de agua parece una tontería pero marca la diferencia",
    "003": "luego bajo a prepararme el desayuno no soy muy de desayunos elaborados la verdad normalmente me hago unas tostadas con aceite y tomate o a veces unos huevos revueltos si tengo más tiempo y café claro que sin café no hay manera de empezar el día",
    "004": "lo del café es un tema serio en mi casa mi madre siempre dice que tomo demasiado que me va a poner nervioso pero yo le digo que con uno o dos cafés al día tampoco es para tanto aunque la verdad es que a veces me tomo tres y no se lo digo",
    "005": "una cosa que he empezado a hacer este año es salir a caminar un rato antes de ponerme a trabajar o estudiar no mucho veinte minutos media hora como mucho pero esos veinte minutos te despejan la cabeza de una forma increíble lo recomiendo a todo el mundo",
    "006": "el barrio donde vivo tiene un parque bastante chulo con árboles grandes y un estanque pequeño donde hay patos y a esa hora de la mañana hay poca gente solo algunos señores mayores dando su paseo y algún que otro corredor es muy tranquilo la verdad",
    "007": "cuando vuelvo a casa ya me siento más despejado y con ganas de hacer cosas me siento en el escritorio abro el ordenador y empiezo a organizar lo que tengo que hacer ese día siempre hago una lista porque si no la hago se me olvida la mitad de las cosas",
    "008": "las listas son algo que me ha cambiado la vida en serio antes iba al supermercado sin lista y siempre olvidaba algo importante llegaba a casa y me daba cuenta de que me había olvidado la leche o el papel de cocina ahora lo apunto todo y funciona mucho mejor",
    "009": "sobre el mediodía suelo hacer una pausa larga para comer en españa eso es bastante normal la comida del mediodía es la más importante del día en mi casa solemos comer juntos cuando podemos y eso mola porque es un momento para charlar y desconectar un poco",
    "010": "mi madre cocina muy bien eso hay que reconocerlo hace un arroz con pollo que está de muerte el secreto según ella es el pimentón ahumado y dejar que el arroz absorba bien el caldo yo lo he intentado hacer solo y nunca me sale igual no sé qué hago mal",
    "011": "por las tardes es cuando más me cuesta concentrarme eso del bajón de después de comer es real no es un mito hay gente que echa la siesta y luego rinde muchísimo más yo lo he intentado pero me cuesta dormirme con luz de día así que prefiero tomar otro café y seguir",
    "012": "cuando termino de trabajar o de estudiar sobre las seis o las siete suelo quedar con amigos o hacer algo de deporte no soy muy de gimnasio la verdad pero sí me gusta jugar al fútbol con los amigos los martes y los jueves llevo ya tres años en ese equipo de barrio",
    "013": "hablando de otra cosa quería decir algo sobre los móviles y las redes sociales porque creo que tengo una relación bastante complicada con todo eso o sea los uso todos los días no voy a mentir pero hay momentos en los que pienso que sería mejor desconectar más",
    "014": "lo que más me engancha son los vídeos cortos esos que duran quince o treinta segundos empiezas a ver uno y de repente han pasado cuarenta minutos y no sabes ni cómo es una locura el algoritmo está diseñado para que no puedas parar y la verdad es que funciona demasiado bien",
    "015": "he intentado ponerme límites de tiempo en las aplicaciones y el móvil me avisa cuando llevo mucho rato pero claro cuando te avisa le das a ignorar y sigues así que el aviso no sirve de mucho si no tienes la fuerza de voluntad para hacerle caso",
    "016": "sin embargo también creo que las redes tienen cosas muy positivas por ejemplo he aprendido un montón de cosas viendo vídeos de gente que explica temas de historia ciencia o cocina hay creadores de contenido que son increíblemente buenos explicando cosas complicadas de manera sencilla",
    "017": "también me parece útil para estar en contacto con gente que vive lejos tengo un amigo que se fue a vivir a berlín hace dos años y gracias a los grupos de whatsapp y a las videollamadas parece que no hubiera tanta distancia antes eso era mucho más difícil",
    "018": "lo que sí me molesta bastante es la cantidad de información falsa que circula la gente comparte cosas sin verificar si son verdad y luego eso se extiende como la pólvora mi tío es un experto en enviar audios de whatsapp con noticias falsas le hemos dicho mil veces que compruebe antes de reenviar",
    "019": "el tema de la privacidad también me preocupa aunque reconozco que no hago gran cosa al respecto acepto los términos y condiciones sin leerlos como todo el mundo sé que esas aplicaciones recopilan un montón de datos sobre mí pero claro la comodidad puede más que la preocupación",
    "020": "una cosa que me parece fascinante es cómo ha cambiado la forma de comunicarse en los últimos diez o quince años mis padres se escribían cartas cuando eran jóvenes cartas de papel con sello y todo ahora mismo mandar una carta física parece algo casi exótico",
    "021": "también ha cambiado mucho el lenguaje usamos abreviaturas emojis memes para expresar cosas que antes necesitaban una frase entera no sé si eso empobrece el idioma o simplemente lo adapta supongo que cada generación transforma el lenguaje a su manera y siempre ha sido así",
    "022": "tengo amigos que dicen que se quieren bajar de todas las redes sociales y llevan años diciéndolo sin hacerlo yo entiendo la idea pero tampoco lo veo realista para la mayoría de la gente hoy en día para bien o para mal estar en las redes es casi una necesidad social",
    "023": "bueno cambiando de tema quería hablar de los libros porque últimamente he retomado el hábito de leer y la verdad es que me alegro mucho hubo una época en que leía bastante luego lo dejé casi por completo y ahora estoy intentando volver a meterme en ello",
    "024": "el mes pasado fui de viaje con unos amigos a portugal concretamente a oporto y fue una experiencia genial hacía tiempo que teníamos ganas de hacer ese viaje y al final cuadramos las fechas y nos fuimos cuatro días éramos cinco en total yo carlos marta luis y ana",
    "025": "fuimos en coche que desde donde vivimos son unas cuatro horas aproximadamente salimos el viernes por la mañana temprano para aprovechar el día carlos se empeñó en conducir él todo el camino de ida que es muy suyo eso no le gusta que nadie más conduzca cuando sale con él",
    "026": "oporto me pareció una ciudad preciosa de verdad tiene una mezcla muy interesante de cosas antiguas y modernas los edificios con azulejos de colores los tranvías el río duero con los barcos es fotogénico hasta decir basta cada esquina parece una postal",
    "027": "lo primero que hicimos al llegar fue buscar el apartamento que habíamos alquilado estaba en el centro histórico en un edificio antiguo con las escaleras un poco inclinadas y las paredes llenas de azulejos era pequeño pero tenía mucho encanto y la ubicación era perfecta",
    "028": "esa primera tarde la usamos para explorar el barrio a pie sin ningún plan en concreto que eso es lo mejor de viajar en mi opinión perderte un poco sin destino fijo entrar en las calles más pequeñas descubrir una plaza que no estaba en ninguna guía eso no tiene precio",
    "029": "por la noche fuimos a cenar a un restaurante que nos recomendó la dueña del apartamento comimos bacalao de tres formas distintas porque en portugal el bacalao es casi una religión dicen que tienen más de trescientas recetas diferentes con bacalao no sé si es verdad pero me lo creo",
    "030": "el sábado hicimos una excursión a la bodega sandeman que está al otro lado del río en vila nova de gaia allí te explican cómo se hace el vino de oporto la diferencia entre los distintos tipos cómo envejece en las barricas y al final hay una cata que es la parte que más nos gustó",
    "031": "marta no bebe alcohol así que ella estuvo tomando notas en su cuaderno y haciendo fotos de todo siempre lleva un cuaderno cuando viaja y lo llena de dibujos notas entradas de museos y esas cosas dice que es su forma de recordar los viajes y la verdad es que tiene mucha razón",
    "032": "el domingo por la mañana madrugamos para ver el mercado de bolhão que es un mercado cubierto histórico que acaban de renovar había frutas verduras flores pescado fresco especias el olor cuando entras es increíble compramos unas pastas de nata para desayunar allí mismo",
    "033": "las pastas de nata son uno de los grandes inventos de la humanidad lo digo en serio son esos pastelitos de hojaldre con crema que se comen templados con un poco de canela por encima solos ya están buenísimos pero con un café cortado son la combinación perfecta",
    "034": "la última tarde ya antes de volver subimos a la torre dos clérigos para ver la ciudad desde arriba son muchos escalones bastante empinados y estrechos pero merece la pena completamente desde arriba se ve toda la ciudad el río los tejados es una vista espectacular",
    "035": "el viaje de vuelta fue más tranquilo porque todos íbamos cansados pero contentos pusimos música hablamos del viaje de lo que más nos había gustado de lo que haríamos diferente la próxima vez ese momento en el coche de vuelta tiene algo especial como cuando asimilas todo lo vivido",
    "036": "pensando en ese viaje me doy cuenta de que tengo que viajar más es algo que siempre digo pero que luego se me pasa cuando la rutina me absorbe creo que merece la pena hacer el esfuerzo de organizar esas escapadas aunque sea de vez en cuando aunque sea cerca",
    "037": "no hace falta ir muy lejos para desconectar eso también lo he aprendido a veces con pasar un fin de semana en un pueblo tranquilo sin mucho plan ya te recargas las pilas el problema es que en el día a día cuesta mucho parar y decir esta semana lo hago",
    "038": "otro tema que me ronda la cabeza últimamente es el de los idiomas llevo años queriendo aprender portugués en serio y después del viaje a oporto todavía más entendía bastante pero me costaba responder con fluidez tengo que buscar algún método que me funcione y ser constante",
    "039": "con el inglés me pasó lo mismo estuve muchos años estudiándolo en clase sin aprender de verdad y lo que me cambió fue ver series y películas en versión original con subtítulos en inglés al cabo de unos meses empecé a entender mucho más y a hablar con menos miedo",
    "040": "el miedo a equivocarse es el mayor obstáculo cuando aprendes un idioma creo yo mucha gente sabe gramática perfectamente pero no habla porque le da vergüenza cometer errores y la única forma de mejorar es hablar equivocarte corregirte y volver a intentarlo no hay otra",
    "041": "hablando de aprender cosas este año me he apuntado a un taller de fotografía y estoy aprendiendo un montón antes hacía fotos de manera intuitiva sin saber nada de encuadre de luz de composición ahora me fijo en cosas que antes ni veía y disfruto mucho más del proceso",
    "042": "lo que más me ha sorprendido del taller es que no hace falta tener una cámara carísima para hacer buenas fotos el profesor dice que la mejor cámara es la que tienes encima que puede ser perfectamente el móvil si sabes usarlo bien la técnica y la mirada importan más que el equipo",
    "043": "en el taller somos diez personas de edades muy distintas hay una señora de sesenta y pico años que lleva toda la vida queriendo aprender fotografía y por fin se ha animado tiene una energía increíble y unas ganas de aprender que dan gusto a mí me parece muy inspirador",
    "044": "una cosa que me he propuesto para los próximos meses es intentar reducir el estrés sé que todo el mundo lo dice pero yo creo que me lo tomo demasiado en serio todo pequeñas cosas que salen mal me afectan más de lo que deberían tengo que aprender a relativizar más",
    "045": "un amigo mío lleva un año meditando por las mañanas solo diez minutos al día y dice que le ha cambiado bastante la forma de reaccionar ante las cosas yo lo he intentado un par de veces pero me distraigo enseguida quizás debería darle otra oportunidad con más paciencia",
    "046": "también quiero leer más este año ya lo dije antes pero especialmente quiero leer más cosas que estén fuera de mi zona de confort novela histórica ensayo poesía cosas que normalmente no elegiría creo que eso ensancha la perspectiva y hace que te enfrentes a las cosas de otra manera",
    "047": "el último libro que leí me lo recomendó mi hermana era una novela de una autora colombiana que no conocía de nada y resultó ser de las mejores que he leído en mucho tiempo eso es lo bueno de las recomendaciones de personas que te conocen bien suelen acertar",
    "048": "bueno creo que ya he hablado bastante por hoy la verdad es que cuando empiezo a hablar de estas cosas me cuesta parar se me ocurren mil cosas que decir espero que haya sido interesante o por lo menos entretenido cada uno tiene sus temas y sus manías ¿no?",
    "049": "si tuviera que quedarme con una idea de todo lo que he contado hoy creo que sería esta vale la pena prestar atención a las cosas pequeñas las mañanas tranquilas un buen desayuno una conversación con un amigo un viaje bien aprovechado eso es lo que luego recuerdas",
    "050": "no sé a veces nos obsesionamos con los grandes objetivos y nos olvidamos de disfrutar el camino yo intento acordarme de eso aunque no siempre lo consigo pero bueno eso es lo bonito también ¿no? que siempre hay algo en lo que mejorar hasta la próxima",
}

META_PATH = Path("data/personal/metadata.csv")
META_PATH.parent.mkdir(parents=True, exist_ok=True)
Path("data/personal/audio").mkdir(parents=True, exist_ok=True)

rows = []
for chunk_id, transcript in CHUNKS.items():
    rows.append({
        "filename": f"SPEAKER_{chunk_id}.wav",
        "chunk": chunk_id,
        "speaker": "SPEAKER",
        "difficulty": 1,
        "transcript": transcript,
        "notes": "",
    })

df = pd.DataFrame(rows)
df.to_csv(META_PATH, index=False)
print(f"Created {META_PATH} with {len(df)} rows.")
print()
print("Next steps:")
print("1. For each speaker, duplicate the rows and replace SPEAKER with the speaker name")
print("     e.g. pedro_001.wav, ana_001.wav - same chunk = same transcript")
print("2. Set difficulty: 1=clear, 2=casual, 3=fast/noisy")
print("3. Place audio files in data/personal/audio/")
print("4. Run: conda run -n whisper-asr python scripts/03_personal_data.py")
