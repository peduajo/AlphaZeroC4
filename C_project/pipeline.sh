#!/bin/bash

should_exit=false

# Función para matar procesos en segundo plano
cleanup() {
    echo "Matando procesos en segundo plano..."
    kill $(jobs -p) 2>/dev/null
    should_exit=true
}

# Configurar trap para llamar a cleanup cuando se reciba SIGINT (Ctrl+C)
trap cleanup SIGINT

cd build

cmake --build .

NUM_PAR=2
NUM_ITER=500

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num-par) NUM_PAR="$2"; shift ;;
        --num-iter) NUM_ITER="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

for ((iteration=0; iteration<NUM_ITER; iteration++))
do
    for ((idx_process=0; idx_process<NUM_PAR; idx_process++))
    do
        ./alphazero_train "${iteration}" "${idx_process}" &
        sleep 1
    done

    wait

    # Verificar si se debe salir del script después de manejar SIGINT
    if [ "$should_exit" = true ] ; then
        echo "Saliendo debido a interrupción del usuario..."
        exit
    fi

    # Realizar la petición CURL y capturar la respuesta
    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"data_path\": \"C_project/data/${iteration}/\"}")

    # Comprobar si el código de respuesta es 200
    if [ "$response" -eq 200 ]; then
        echo "Petición CURL exitosa para la iteración ${iteration}."
    else
        echo "Error en la petición CURL para la iteración ${iteration}. Código de respuesta: ${response}."
    fi
done
