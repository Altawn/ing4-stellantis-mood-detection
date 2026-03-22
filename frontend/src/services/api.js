import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const api = {
  async sendFrame(imageBase64, temperature) {
    try {
      const response = await axios.post(`${API_BASE_URL}/frame`, {
        image: imageBase64
        // temperature: temperature // ignored by backend now
      });
      return response.data;
    } catch (error) {
      console.error('Erreur envoi frame:', error);
      throw error;
    }
  },

  async changeTemperature(delta) {
    try {
      const response = await axios.post(`${API_BASE_URL}/temperature`, {
        delta: delta
      });
      return response.data;
    } catch (error) {
      console.error('Erreur changement temperature:', error);
      throw error;
    }
  },

  async checkVLM() {
    try {
      const response = await axios.get(`${API_BASE_URL}/vlm-check`);
      return response.data;
    } catch (error) {
      console.error('Erreur VLM check:', error);
      throw error;
    }
  },

  async sendVLMResponse(userResponse) {
    try {
      const response = await axios.post(`${API_BASE_URL}/vlm-response`, {
        response: userResponse
      });
      return response.data;
    } catch (error) {
      console.error('Erreur VLM response:', error);
      throw error;
    }
  }
};
