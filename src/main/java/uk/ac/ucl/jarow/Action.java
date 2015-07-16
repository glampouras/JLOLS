package uk.ac.ucl.jdagger.jarow;

public class Action {
    private String decision;
    
    public Action(String decision) {
        this.decision = decision;
    }

    public String getDecision() {
        return decision;
    }

    public void setDecision(String decision) {
        this.decision = decision;
    }
}
