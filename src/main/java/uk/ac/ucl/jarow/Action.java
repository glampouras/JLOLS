package uk.ac.ucl.jarow;

public class Action {
    private String decision;
    
    public Action(String decision) {
        this.decision = decision;
    }
    
    public Action(Action a) {
        this.decision = a.getDecision();
    }

    public String getDecision() {
        return decision;
    }

    public void setDecision(String decision) {
        this.decision = decision;
    }

    @Override
    public String toString() {
        return "A{" + decision + '}';
    }
}
