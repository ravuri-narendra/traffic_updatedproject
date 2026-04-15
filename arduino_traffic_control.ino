#define N_RED 2
#define N_YELLOW 4
#define N_GREEN 5

#define S_RED 18
#define S_YELLOW 19
#define S_GREEN 21

#define E_RED 22
#define E_YELLOW 23
#define E_GREEN 25

#define W_RED 26
#define W_YELLOW 27
#define W_GREEN 14

char currentDir = 0;
char currentColor = 0;
unsigned long lastBlink = 0;
bool blinkState = false;

void allOff() {
  digitalWrite(N_RED, LOW); digitalWrite(N_YELLOW, LOW); digitalWrite(N_GREEN, LOW);
  digitalWrite(S_RED, LOW); digitalWrite(S_YELLOW, LOW); digitalWrite(S_GREEN, LOW);
  digitalWrite(E_RED, LOW); digitalWrite(E_YELLOW, LOW); digitalWrite(E_GREEN, LOW);
  digitalWrite(W_RED, LOW); digitalWrite(W_YELLOW, LOW); digitalWrite(W_GREEN, LOW);
}

void allRed() {
  allOff();
  digitalWrite(N_RED, HIGH);
  digitalWrite(S_RED, HIGH);
  digitalWrite(E_RED, HIGH);
  digitalWrite(W_RED, HIGH);
}

int getYellowPin(char d) {
  if (d == 'N') return N_YELLOW;
  if (d == 'S') return S_YELLOW;
  if (d == 'E') return E_YELLOW;
  if (d == 'W') return W_YELLOW;
  return -1;
}

int getRedPin(char d) {
    if (d == 'N') return N_RED;
    if (d == 'S') return S_RED;
    if (d == 'E') return E_RED;
    if (d == 'W') return W_RED;
    return -1;
}

void setLight(char d, char c) {
  currentDir = d;
  currentColor = c;
  allRed();

  if (c == 'G') {
    int rPin = getRedPin(d);
    int gPin = (d == 'N') ? N_GREEN : (d == 'S') ? S_GREEN : (d == 'E') ? E_GREEN : W_GREEN;
    digitalWrite(rPin, LOW);
    digitalWrite(gPin, HIGH);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(N_RED, OUTPUT); pinMode(N_YELLOW, OUTPUT); pinMode(N_GREEN, OUTPUT);
  pinMode(S_RED, OUTPUT); pinMode(S_YELLOW, OUTPUT); pinMode(S_GREEN, OUTPUT);
  pinMode(E_RED, OUTPUT); pinMode(E_YELLOW, OUTPUT); pinMode(E_GREEN, OUTPUT);
  pinMode(W_RED, OUTPUT); pinMode(W_YELLOW, OUTPUT); pinMode(W_GREEN, OUTPUT);
  allRed();
  Serial.println("Traffic System Ready");
}

void loop() {
  if (Serial.available() >= 3) {
    char d = Serial.read();     
    char c = Serial.read();   
    Serial.read();           
    setLight(d, c);
  }

  // --- FLASHING LOGIC ---
  if (currentColor == 'Y') {
    if (millis() - lastBlink >= 500) { 
      lastBlink = millis();
      blinkState = !blinkState;
      
      int yPin = getYellowPin(currentDir);
      int rPin = getRedPin(currentDir);
      
      digitalWrite(rPin, LOW); 
      digitalWrite(yPin, blinkState ? HIGH : LOW);
    }
  }
}
