import vk_api
import csv
import time

def get_friends_network(user_id, token, user_name='USER'):
    vk_session = vk_api.VkApi(token=token)
    vk = vk_session.get_api()
    
    # Получаем своих друзей
    my_friends = vk.friends.get(user_id=user_id, fields="first_name,last_name,city,education,bdate,has_mobile,relation,sex,timezone,personal")
    
    # Получаем друзей друзей
    network = {}
    for i, friend in enumerate(my_friends["items"]):
        try:
            print(f"Обрабатываю друга {i+1}/{len(my_friends['items'])}: {friend['first_name']} {friend['last_name']}")
            friend_friends = vk.friends.get(user_id=friend["id"], fields="first_name,last_name,city,education,bdate,has_mobile,relation,sex,timezone,personal")
            network[friend["id"]] = {
                "friend_info": friend,
                "friends": friend_friends["items"],
                "handshake_ID": user_id,
                "handshake_name": user_name
            }
            # Задержка чтобы не превысить лимиты API
            time.sleep(0.2)
        except Exception as e:
            print(f"Ошибка при получении друзей пользователя {friend['id']}: {e}")
            continue
    
    return network

def save_network_to_csv(network, filename="friends_network.csv"):
    """Сохраняет сеть друзей в CSV файл"""
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Заголовки CSV
        writer.writerow([
            'ID', 
            'Name', 
            'handshake_ID',
            'handshake_name',
            'Connection Type',
            'city',
            'university',
            'university_name',
            'faculty',
            'faculty_name',
            'graduation_year',
            'bdate',
            'has_mobile',
            'relation',
            'sex',
            'timezone',
            'personal',
        ])
        
        # Записываем данные
        for friend_id, data in network.items():
            friend_info = data["friend_info"]
            friend_name = f"{friend_info.get('first_name', '')} {friend_info.get('last_name', '')}"
            
            # Записываем прямых друзей (первый уровень)
            writer.writerow([
                friend_id,
                friend_name,
                data['handshake_ID'],
                data['handshake_name'],
                'Direct Friend',
                friend_info.get('city', ''),
                friend_info.get('university', ''),
                friend_info.get('university_name', ''),
                friend_info.get('faculty', ''),
                friend_info.get('faculty_name', ''),
                friend_info.get('graduation', ''),
                friend_info.get('bdate', ''),
                friend_info.get('has_mobile', ''),
                friend_info.get('relation', ''),
                friend_info.get('sex', ''),
                friend_info.get('timezone', ''),
                friend_info.get('personal', ''),
            ])
            
            # Записываем друзей друзей (второй уровень)
            for fof in data["friends"]:
                fof_id = fof["id"]
                fof_name = f"{fof.get('first_name', '')} {fof.get('last_name', '')}"
                
                writer.writerow([
                    fof_id,
                    fof_name,
                    friend_id,
                    friend_name,
                    'Friend of Friend',
                    fof.get('city', ''),
                    fof.get('university', ''),
                    fof.get('university_name', ''),
                    fof.get('faculty', ''),
                    fof.get('faculty_name', ''),
                    fof.get('graduation', ''),
                    fof.get('bdate', ''),
                    fof.get('has_mobile', ''),
                    fof.get('relation', ''),
                    fof.get('sex', ''),
                    fof.get('timezone', ''),
                    fof.get('personal', ''),
                    
                ])

def save_network_summary(network, filename="friends_network_summary.csv"):
    """Сохраняет сводную информацию в отдельный CSV"""
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Заголовки для сводной таблицы
        writer.writerow([
            'Friend ID',
            'First Name',
            'Last Name',
            'Number of Friends',
            'Friends IDs',
            'Friends Names'
        ])
        
        for friend_id, data in network.items():
            friend_info = data["friend_info"]
            friends_list = data["friends"]
            
            # Собираем ID друзей
            friend_ids = [str(f["id"]) for f in friends_list]
            friend_names = [f"{f.get('first_name', '')} {f.get('last_name', '')}" for f in friends_list]
            
            writer.writerow([
                friend_id,
                friend_info.get('first_name', ''),
                friend_info.get('last_name', ''),
                len(friends_list),
                '; '.join(friend_ids),
                '; '.join(friend_names)
            ])
        

# Основной код выполнения
if __name__ == "__main__":
    # Ваши данные
    ACCESS_TOKEN = 'c5c969a8c5c969a8c5c969a8c1c6f2a2ebcc5c9c5c969a8ad3d97fcc2886962cfe15292'
    big_network = {}
    networks_list = []
    for user_id, user_name in [(273517530, 'EGOR K'), (232453448, 'ALEX S'), (184300739, "EGOR S")]:
    #for user_id, user_name in [(347120445, 'TEST U')]:
        
        try:
            # Получаем сеть друзей
            print("Начинаю сбор данных о друзьях...")
            network = get_friends_network(user_id, ACCESS_TOKEN, user_name)
            #big_network.update(network)
            networks_list.append(network)
            # Сохраняем в CSV
            save_network_to_csv(network, str(user_id)+"_friends_network_detailed.csv")
            save_network_summary(network, str(user_id)+"_friends_network_summary.csv")
            
            # Статистика
            total_direct_friends = len(network)
            total_friends_of_friends = sum(len(data["friends"]) for data in network.values())
            unique_friends_of_friends = set()
            
            for data in network.values():
                for friend in data["friends"]:
                    unique_friends_of_friends.add(friend["id"])
            
            print(f"\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Прямых друзей: {total_direct_friends}")
            print(f"Всего связей друзей друзей: {total_friends_of_friends}")
            print(f"Уникальных друзей друзей: {len(unique_friends_of_friends)}")
            print(f"Данные сохранены в файлы:")
            print(f"- friends_network_detailed.csv (детальная информация)")
            print(f"- friends_network_summary.csv (сводная информация)")
            
        except Exception as e:
            print(f"Произошла ошибка: {e}")
    for network in networks_list:
        big_network.update(network)
    save_network_to_csv(big_network, "friends_big_network_detailed.csv")
    save_network_summary(big_network, "friends_big_network_summary.csv")
