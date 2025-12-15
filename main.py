import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class KompotEnvironment:
    """Среда для задачи распределения компота"""
    
    def __init__(self):
        self.num_glasses = 5
        self.max_volume = 100  # максимальный объем в каждом стакане
        self.target_level = 50  # целевой уровень в каждом стакане
        
        # Начальное состояние - все стаканы пустые
        self.initial_state = tuple([0] * self.num_glasses)
        self.reset()
    
    def reset(self):
        """Сброс среды к начальному состоянию"""
        self.glasses = list(self.initial_state)
        self.pickups = 0
        self.total_poured = 0
        self.done = False
        return self.get_state()
    
    def get_state(self):
        """Получение текущего состояния как кортежа"""
        return tuple(self.glasses)
    
    def step(self, action):
        """
        Выполнение действия агентом
        action: (glass_index, amount) - налить amount в стакан glass_index
        amount > 0: налить из кувшина
        amount < 0: вылить из стакана
        """
        glass_idx, amount = action
        
        # Если пытаемся налить/вылить, это считается подъемом стакана
        if amount != 0:
            self.pickups += 1
        
        # Ограничиваем изменение объема
        new_volume = self.glasses[glass_idx] + amount
        new_volume = max(0, min(self.max_volume, new_volume))
        
        # Обновляем объем
        self.glasses[glass_idx] = new_volume
        self.total_poured += abs(amount)
        
        # Вычисляем награду
        reward = self.calculate_reward()
        
        # Проверяем, достигнута ли цель
        if self.is_target_reached():
            self.done = True
            # Дополнительная награда за завершение
            reward += 100
            
        return self.get_state(), reward, self.done, self.pickups
    
    def calculate_reward(self):
        """Вычисление награды"""
        # Основная награда: отрицательная за отклонение от цели
        deviation_penalty = -sum(abs(level - self.target_level) for level in self.glasses) / 100
        
        # Штраф за каждое поднятие стакана
        pickup_penalty = -self.pickups * 10
        
        # Штраф за общее количество налитого (чтобы минимизировать лишние действия)
        pour_penalty = -self.total_poured / 500
        
        return deviation_penalty + pickup_penalty + pour_penalty
    
    def is_target_reached(self):
        """Проверка, достигнута ли цель"""
        tolerance = 5  # допуск
        return all(abs(level - self.target_level) <= tolerance for level in self.glasses)
    
    def render(self):
        """Визуализация текущего состояния"""
        print(f"Стаканы: {self.glasses}")
        print(f"Поднятий: {self.pickups}, Всего налито: {self.total_poured}")
        print("-" * 40)


class KompotAgent:
    """Агент для распределения компота"""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        
        # Q-table: состояние -> действие -> значение
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Возможные действия
        self.actions = []
        for glass in range(env.num_glasses):
            for amount in [-20, -10, -5, 5, 10, 20]:
                self.actions.append((glass, amount))
        
        # Для отслеживания обучения
        self.rewards_history = []
        self.pickups_history = []
    
    def get_q_value(self, state, action):
        """Получение Q-значения для состояния и действия"""
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        """Обновление Q-значения по правилу Q-learning"""
        # Максимальное Q-значение для следующего состояния
        next_max = max([self.get_q_value(next_state, a) for a in self.actions]) if next_state in self.q_table else 0
        
        # Текущее Q-значение
        current_q = self.get_q_value(state, action)
        
        # Новое Q-значение
        new_q = current_q + self.lr * (reward + self.gamma * next_max - current_q)
        
        self.q_table[state][action] = new_q
    
    def choose_action(self, state):
        """Выбор действия с использованием ε-жадной стратегии"""
        if random.random() < self.epsilon:
            # Исследование: случайное действие
            return random.choice(self.actions)
        else:
            # Использование: лучшее действие
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            max_q = max(q_values.values())
            
            # Выбираем все действия с максимальным Q-значением
            best_actions = [action for action, q in q_values.items() if q == max_q]
            
            return random.choice(best_actions) if best_actions else random.choice(self.actions)
    
    def decay_exploration(self):
        """Уменьшение вероятности исследования"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes=1000):
        """Обучение агента"""
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            episode_pickups = 0
            
            while not self.env.done:
                # Выбор действия
                action = self.choose_action(state)
                
                # Выполнение действия
                next_state, reward, done, pickups = self.env.step(action)
                episode_pickups = pickups
                
                # Обновление Q-значения
                self.update_q_value(state, action, reward, next_state)
                
                # Переход к следующему состоянию
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Уменьшение вероятности исследования
            self.decay_exploration()
            
            # Сохранение истории
            self.rewards_history.append(total_reward)
            self.pickups_history.append(episode_pickups)
            
            # Вывод прогресса
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                avg_pickups = np.mean(self.pickups_history[-100:])
                print(f"Эпизод {episode + 1}/{episodes}, "
                      f"Ср. награда: {avg_reward:.2f}, "
                      f"Ср. поднятий: {avg_pickups:.1f}, "
                      f"ε: {self.epsilon:.3f}")
    
    def test(self, max_steps=50):
        """Тестирование обученного агента"""
        state = self.env.reset()
        steps = 0
        
        print("\n" + "="*50)
        print("ТЕСТИРОВАНИЕ ОБУЧЕННОГО АГЕНТА")
        print("="*50)
        
        self.env.render()
        
        while not self.env.done and steps < max_steps:
            # Выбор лучшего действия (без исследования)
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            action = max(q_values.items(), key=lambda x: x[1])[0]
            
            # Выполнение действия
            state, reward, done, pickups = self.env.step(action)
            
            print(f"Шаг {steps + 1}: Действие {action}")
            self.env.render()
            steps += 1
        
        print(f"\nИтог: достигнуто состояние {self.env.glasses}")
        print(f"Поднятий стаканов: {self.env.pickups}")
        print(f"Отклонение от цели: {sum(abs(l - self.env.target_level) for l in self.env.glasses)}")


def plot_learning_curve(rewards_history, pickups_history):
    """Построение кривых обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Сглаживание кривых
    window_size = 50
    smoothed_rewards = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
    smoothed_pickups = np.convolve(pickups_history, np.ones(window_size)/window_size, mode='valid')
    
    # График наград
    ax1.plot(rewards_history, alpha=0.3, label='Сырые данные')
    ax1.plot(range(window_size-1, len(rewards_history)), smoothed_rewards, 
             label=f'Скользящее среднее ({window_size} эпизодов)', linewidth=2)
    ax1.set_xlabel('Номер эпизода')
    ax1.set_ylabel('Награда')
    ax1.set_title('Кривая обучения: Награда агента')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График поднятий стаканов
    ax2.plot(pickups_history, alpha=0.3, label='Сырые данные')
    ax2.plot(range(window_size-1, len(pickups_history)), smoothed_pickups,
             label=f'Скользящее среднее ({window_size} эпизодов)', linewidth=2)
    ax2.set_xlabel('Номер эпизода')
    ax2.set_ylabel('Количество поднятий')
    ax2.set_title('Кривая обучения: Поднятия стаканов')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    env = KompotEnvironment()
    agent = KompotAgent(env, learning_rate=0.1, discount_factor=0.95,
                       exploration_rate=1.0, exploration_decay=0.998)
    
    print("Начало обучения...")
    agent.train(episodes=2000)
    
    # Тестирование
    agent.test()
    
    # Визуализация кривых обучения
    plot_learning_curve(agent.rewards_history, agent.pickups_history)
    
    print("\n" + "="*50)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*50)
    print(f"Всего состояний в Q-table: {len(agent.q_table)}")
    print(f"Последняя ε (вероятность исследования): {agent.epsilon:.4f}")
    print(f"Средняя награда за последние 100 эпизодов: {np.mean(agent.rewards_history[-100:]):.2f}")
    print(f"Среднее количество поднятий за последние 100 эпизодов: {np.mean(agent.pickups_history[-100:]):.1f}")


if __name__ == "__main__":
    main()