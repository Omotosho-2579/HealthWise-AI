from typing import Dict, List
import random

class WellnessRecommender:
    """
    Provides personalized wellness recommendations based on user profile.
    
    Uses content-based filtering to match user goals with wellness tips.
    """
    
    def __init__(self, wellness_tips: List[Dict]):
        """
        Initialize recommender with wellness tips database.
        
        Args:
            wellness_tips: List of wellness tip dictionaries
        """
        self.wellness_tips = wellness_tips
    
    def get_recommendation(self, user_profile: Dict) -> Dict:
        """
        Get personalized wellness recommendation.
        
        Args:
            user_profile: User's health profile containing goals
            
        Returns:
            Recommended wellness tip with explanation
        """
        health_goals = user_profile.get('health_goals', ['general_wellness'])
        
        # Filter tips matching user's goals
        matching_tips = []
        for tip in self.wellness_tips:
            tip_goals = tip.get('health_goals', [])
            if any(goal in tip_goals for goal in health_goals):
                matching_tips.append(tip)
        
        # If no matches, return random tip
        if not matching_tips:
            matching_tips = self.wellness_tips
        
        # Select best matching tip
        selected_tip = random.choice(matching_tips)
        
        return {
            'tip': selected_tip,
            'reason': f"This tip aligns with your health goals: {', '.join(health_goals)}"
        }
    
    def check_proactive_nudge(self, user_data: Dict) -> Tuple[bool, str]:
        """
        Check if user needs a proactive health nudge.
        
        Args:
            user_data: User's recent health data
            
        Returns:
            Tuple of (should_nudge, nudge_type)
        """
        # Rule 1: Poor sleep pattern
        sleep_hours = user_data.get('avg_sleep_hours', 7)
        if sleep_hours < 6:
            return True, "sleep_hygiene"
        
        # Rule 2: High stress indicators
        stress_level = user_data.get('stress_level', 'low')
        if stress_level == 'high':
            return True, "stress_management"
        
        # Rule 3: Low activity
        daily_steps = user_data.get('daily_steps', 5000)
        if daily_steps < 3000:
            return True, "physical_activity"
        
        return False, None